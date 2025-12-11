"""
Hybrid VMD-TMFG-LSTM + XGBoost Model Architecture for MANTIS Mining
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
import pickle
import json

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    
    # Configure GPU usage with optimizations
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Enable mixed precision for faster training on modern GPUs
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print(f"✓ GPU detected: {len(gpus)} device(s) available")
                print(f"  Using GPU: {gpus[0].name}")
                print(f"  Mixed precision training: ENABLED (faster training)")
            except Exception as e:
                print(f"✓ GPU detected: {len(gpus)} device(s) available")
                print(f"  Using GPU: {gpus[0].name}")
                print(f"  Mixed precision: Not available ({e})")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("ℹ No GPU detected, using CPU")
except ImportError:
    print("Warning: TensorFlow not installed. Install with: pip install tensorflow")
    tf = None
    Sequential = None

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from scripts.feature_engineering.feature_extractor import FeatureExtractor


class VMDTMFGLSTMXGBoost:
    """
    Hybrid VMD-TMFG-LSTM + XGBoost model for MANTIS embeddings
    
    Architecture:
    1. VMD decomposition of prices
    2. TMFG feature selection (top 10 features)
    3. LSTM for temporal modeling (embeddings)
    4. XGBoost for final embedding generation
    """
    
    def __init__(
        self,
        embedding_dim: int = 2,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        time_steps: int = 20,
        vmd_k: int = 8,
        tmfg_n_features: int = 10,
        dropout: float = 0.2,
        learning_rate: float = 0.0005,
        use_gpu: bool = True
    ):
        """
        Initialize model
        
        Args:
            embedding_dim: Output embedding dimension (from config.CHALLENGES)
            lstm_hidden: LSTM hidden units
            lstm_layers: Number of LSTM layers
            time_steps: Sequence length for LSTM
            vmd_k: Number of VMD components
            tmfg_n_features: Number of selected features
            dropout: Dropout rate
        """
        self.embedding_dim = embedding_dim
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.time_steps = time_steps
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.use_gpu = use_gpu
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            vmd_k=vmd_k,
            tmfg_n_features=tmfg_n_features
        )
        
        # Log device info
        if tf is not None:
            if self.use_gpu and tf.config.list_physical_devices('GPU'):
                print(f"  Model will use GPU: {tf.config.list_physical_devices('GPU')[0].name}")
            else:
                print("  Model will use CPU")
        
        # Models
        self.lstm_model = None
        self.xgb_model = None
        self.scaler = StandardScaler()
        self.selected_feature_indices = None
        
        # Training history
        self.training_history = {}
    
    def build_lstm(self, input_dim: int) -> Optional[Model]:
        """
        Build LSTM model for temporal embeddings
        
        Args:
            input_dim: Input feature dimension
            
        Returns:
            Keras Model
        """
        if tf is None or Sequential is None:
            print("TensorFlow not available, cannot build LSTM")
            return None
        
        model = Sequential()
        
        # First LSTM layer (return sequences if multiple layers)
        model.add(LSTM(
            self.lstm_hidden,
            return_sequences=(self.lstm_layers > 1),
            input_shape=(self.time_steps, input_dim)
        ))
        model.add(Dropout(self.dropout))
        
        # Additional LSTM layers
        for i in range(1, self.lstm_layers):
            return_seq = (i < self.lstm_layers - 1)
            model.add(LSTM(self.lstm_hidden, return_sequences=return_seq))
            model.add(Dropout(self.dropout))
        
        # Embedding layer with more capacity for diverse representations
        # This helps capture unique patterns that others might miss
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(self.dropout))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(self.dropout * 0.5))  # Less dropout in deeper layers
        model.add(Dense(32, activation='relu'))
        
        # Output layer (for training - will be replaced for inference)
        # Use float32 for output layer when using mixed precision
        model.add(Dense(1, activation='linear', name='lstm_output', dtype='float32'))
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences for LSTM
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,) or None
            
        Returns:
            Tuple of (X_seq, y_seq) where X_seq is (n_sequences, time_steps, n_features)
        """
        X_seq = []
        y_seq = [] if y is not None else None
        
        for i in range(self.time_steps, len(X)):
            X_seq.append(X[i - self.time_steps:i])
            if y is not None:
                y_seq.append(y[i])
        
        X_seq = np.array(X_seq)
        if y_seq is not None:
            y_seq = np.array(y_seq)
        
        return X_seq, y_seq
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        funding_df: Optional[pd.DataFrame] = None,
        oi_df: Optional[pd.DataFrame] = None,
        cross_exchange_df: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """
        Prepare data for training/inference
        
        Args:
            df: OHLCV DataFrame
            funding_df: Funding rates DataFrame
            oi_df: Open interest DataFrame
            cross_exchange_df: Cross-exchange data (optional)
            
        Returns:
            Tuple of (X_seq, y, feature_names)
        """
        # Extract all features
        df_features = self.feature_extractor.extract_all_features(
            df, funding_df, oi_df
        )
        
        # Prepare feature matrix
        X, y, feature_names = self.feature_extractor.prepare_feature_matrix(
            df_features, target_col='close'
        )
        
        # FIX: Convert absolute prices to price changes (forward-looking)
        # y currently contains absolute close prices, we need price changes
        # For sequence at index i (using data from [i-time_steps:i]), 
        # we want to predict the price change from time i to i+1
        if len(y) > 0:
            # Calculate forward-looking price change: price[i+1] - price[i]
            # This is what we want to predict: will price go up or down?
            y_changes = np.zeros_like(y, dtype=np.float64)
            y_changes[:-1] = np.diff(y)  # next_price - current_price for all but last
            y_changes[-1] = 0  # Last value has no next price, set to 0
            y = y_changes  # Use price changes instead of absolute prices
        else:
            y = np.array([])
        
        # Scale features
        if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # TMFG feature selection (if not already done)
        if self.selected_feature_indices is None:
            # Use price change as target for feature selection
            y_selection = df_features['close'].pct_change().fillna(0).values
            X_selected, selected_indices = self.feature_extractor.tmfg_feature_selection(
                X_scaled, y_selection
            )
            self.selected_feature_indices = selected_indices
        else:
            X_selected = X_scaled[:, self.selected_feature_indices]
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_selected, y)
        
        # Update feature names
        selected_feature_names = [feature_names[i] for i in self.selected_feature_indices]
        
        return X_seq, y_seq, selected_feature_names
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 64,
        verbose: int = 1
    ):
        """
        Train the hybrid model
        
        Args:
            X_train: Training sequences (n_samples, time_steps, n_features)
            y_train: Training targets (n_samples,)
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            epochs: Number of epochs
            batch_size: Batch size
            verbose: Verbosity level
        """
        if tf is None:
            print("TensorFlow not available, cannot train LSTM")
            return
        
        # Build LSTM if not already built
        if self.lstm_model is None:
            input_dim = X_train.shape[2]
            self.lstm_model = self.build_lstm(input_dim)
        
        # Prepare targets (price change for regression)
        y_train_delta = y_train
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=15,
                restore_best_weights=True,
                verbose=verbose
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=verbose
            )
        ]
        
        # Train LSTM
        history = self.lstm_model.fit(
            X_train,
            y_train_delta,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.training_history['lstm'] = history.history
        
        # Extract LSTM embeddings
        # After early stopping restores weights, ensure model is built by doing a forward pass
        _ = self.lstm_model.predict(X_train[:1], verbose=0)
        
        # Create embedding model by removing the last layer
        # In newer Keras versions, use .inputs (plural), fallback to .input (singular)
        if hasattr(self.lstm_model, 'inputs') and self.lstm_model.inputs:
            model_inputs = self.lstm_model.inputs[0] if isinstance(self.lstm_model.inputs, list) else self.lstm_model.inputs
        elif hasattr(self.lstm_model, 'input'):
            model_inputs = self.lstm_model.input
        else:
            # Fallback: get input from first layer
            model_inputs = self.lstm_model.layers[0].input
        
        # Create embedding model
        embedding_model = Model(
            inputs=model_inputs,
            outputs=self.lstm_model.layers[-2].output  # Second-to-last layer (embedding layer)
        )
        
        lstm_embeddings_train = embedding_model.predict(X_train, verbose=0)
        lstm_embeddings_val = embedding_model.predict(X_val, verbose=0) if X_val is not None else None
        
        # Train XGBoost on LSTM embeddings
        print("Training XGBoost on LSTM embeddings...")
        
        # Prepare XGBoost targets (binary direction with threshold)
        # CRITICAL FIX: Use threshold-based labels to balance classes
        # Instead of > 0, use a percentile-based threshold to ensure balance
        threshold = np.percentile(y_train_delta, 50)  # Median split for 50/50 balance
        y_train_binary = (y_train_delta > threshold).astype(int)
        
        if y_val is not None:
            y_val_binary = (y_val > threshold).astype(int)
        else:
            y_val_binary = None
        
        # Calculate class weights for imbalanced data
        # This is critical for high salience - we need to learn minority class patterns
        class_counts = np.bincount(y_train_binary)
        total_samples = len(y_train_binary)
        n_classes = len(class_counts)
        
        if n_classes == 2 and total_samples > 0:
            # Calculate balanced class weights
            class_weights = {}
            for i in range(n_classes):
                if class_counts[i] > 0:
                    # Standard balanced weight: n_samples / (n_classes * count)
                    weight = total_samples / (n_classes * class_counts[i])
                    class_weights[i] = weight
                else:
                    class_weights[i] = 1.0
            
            # Scale weights to sum to n_classes (XGBoost convention)
            weight_sum = sum(class_weights.values())
            if weight_sum > 0:
                for i in class_weights:
                    class_weights[i] = class_weights[i] * n_classes / weight_sum
            
            print(f"  Class distribution: {dict(zip(range(n_classes), class_counts))}")
            print(f"  Class balance: {class_counts[0]/(class_counts[0]+class_counts[1])*100:.1f}% / {class_counts[1]/(class_counts[0]+class_counts[1])*100:.1f}%")
            print(f"  Class weights: {class_weights}")
            print(f"  Threshold used: {threshold:.6f}")
            
            # Create sample weights for XGBoost
            sample_weights = np.array([class_weights[y] for y in y_train_binary])
        else:
            sample_weights = None
        
        # Train XGBoost
        dtrain = xgb.DMatrix(lstm_embeddings_train, label=y_train_binary, weight=sample_weights)
        
        # Determine XGBoost tree method (GPU or CPU)
        tree_method = 'hist'
        xgb_predictor = None
        try:
            from xgboost.core import _has_cuda_support
            has_xgb_gpu = _has_cuda_support()
        except Exception:
            has_xgb_gpu = False
        
        if self.use_gpu and has_xgb_gpu and tf is not None and tf.config.list_physical_devices('GPU'):
            tree_method = 'gpu_hist'
            xgb_predictor = 'gpu_predictor'
            print("  XGBoost will use GPU (gpu_hist)")
        else:
            print("  XGBoost will use CPU (hist method)")
        
        # Calculate base_score from the data (proportion of positive class)
        # This must be in (0,1) for binary:logistic objective
        positive_ratio = y_train_binary.mean()
        base_score = max(0.01, min(0.99, positive_ratio))  # Clamp to (0.01, 0.99)
        
        # Use scale_pos_weight for additional class imbalance handling
        # With median split, this should be ~1.0, but still calculate it
        if n_classes == 2 and class_counts[1] > 0:
            scale_pos_weight = class_counts[0] / class_counts[1]
        else:
            scale_pos_weight = 1.0
        
        print(f"  Base score: {base_score:.4f}, Scale pos weight: {scale_pos_weight:.4f}")
        
        params = {
            'max_depth': 6,
            'eta': 0.1,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'tree_method': tree_method,
            'base_score': base_score,  # Explicitly set base_score for logistic loss
            'scale_pos_weight': scale_pos_weight,  # Handle class imbalance
            'min_child_weight': 1,  # Allow more splits for minority class
            'max_delta_step': 1  # Help with imbalanced data
        }
        if xgb_predictor:
            params['predictor'] = xgb_predictor
        
        evals = [(dtrain, 'train')]
        if lstm_embeddings_val is not None:
            dval = xgb.DMatrix(lstm_embeddings_val, label=y_val_binary)
            evals.append((dval, 'val'))
        
        # Train XGBoost
        try:
            self.xgb_model = xgb.train(
                params,
                dtrain,
                num_boost_round=100,
                evals=evals,
                early_stopping_rounds=10,
                verbose_eval=verbose > 0
            )
        except Exception as e:
            # If training fails for any reason, provide helpful error message
            error_msg = str(e)
            print(f"  XGBoost training error: {error_msg[:200]}")
            raise
        
        # Evaluate
        train_pred = self.xgb_model.predict(dtrain)
        train_pred_binary = (train_pred > 0.5).astype(int)
        train_acc = accuracy_score(y_train_binary, train_pred_binary)
        
        # Debug: Show prediction distribution
        pred_counts = np.bincount(train_pred_binary, minlength=2)
        print(f"  Prediction distribution: Class 0: {pred_counts[0]} ({pred_counts[0]/len(train_pred_binary)*100:.1f}%), Class 1: {pred_counts[1]} ({pred_counts[1]/len(train_pred_binary)*100:.1f}%)")
        print(f"  Prediction mean: {train_pred.mean():.4f} (should be ~0.5 for balanced predictions)")
        
        # Calculate AUC only if we have both classes
        try:
            if len(np.unique(y_train_binary)) > 1 and len(np.unique(train_pred_binary)) > 1:
                train_auc = roc_auc_score(y_train_binary, train_pred)
            else:
                train_auc = float('nan')
                print("  Warning: AUC cannot be calculated (all predictions or labels are the same class)")
        except Exception as e:
            train_auc = float('nan')
            print(f"  Warning: AUC calculation failed: {str(e)[:60]}")
        
        print(f"Train Accuracy: {train_acc:.4f}, AUC: {train_auc:.4f}")
        
        if lstm_embeddings_val is not None:
            val_pred = self.xgb_model.predict(dval)
            val_pred_binary = (val_pred > 0.5).astype(int)
            val_acc = accuracy_score(y_val_binary, val_pred_binary)
            
            # Calculate AUC only if we have both classes
            try:
                if len(np.unique(y_val_binary)) > 1 and len(np.unique(val_pred_binary)) > 1:
                    val_auc = roc_auc_score(y_val_binary, val_pred)
                else:
                    val_auc = float('nan')
                    print("  Warning: Val AUC cannot be calculated (all predictions or labels are the same class)")
            except Exception as e:
                val_auc = float('nan')
                print(f"  Warning: Val AUC calculation failed: {str(e)[:60]}")
            
            print(f"Val Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}")
    
    def predict_embeddings(self, X_seq: np.ndarray) -> np.ndarray:
        """
        Generate MANTIS embeddings
        
        Args:
            X_seq: Input sequences (n_samples, time_steps, n_features)
            
        Returns:
            Embeddings array (n_samples, embedding_dim)
        """
        if self.lstm_model is None or self.xgb_model is None:
            # Return zero embeddings if model not trained
            return np.zeros((len(X_seq), self.embedding_dim))
        
        # Ensure model is built by doing a forward pass
        if len(X_seq) > 0:
            _ = self.lstm_model.predict(X_seq[:1], verbose=0)
        
        # Extract LSTM embeddings
        # In newer Keras versions, use .inputs (plural), fallback to .input (singular)
        if hasattr(self.lstm_model, 'inputs') and self.lstm_model.inputs:
            model_inputs = self.lstm_model.inputs[0] if isinstance(self.lstm_model.inputs, list) else self.lstm_model.inputs
        elif hasattr(self.lstm_model, 'input'):
            model_inputs = self.lstm_model.input
        else:
            # Fallback: get input from first layer
            model_inputs = self.lstm_model.layers[0].input
        
        embedding_model = Model(
            inputs=model_inputs,
            outputs=self.lstm_model.layers[-2].output
        )
        lstm_embeddings = embedding_model.predict(X_seq, verbose=0)
        
        # Get XGBoost predictions (probabilities)
        dtest = xgb.DMatrix(lstm_embeddings)
        xgb_pred = self.xgb_model.predict(dtest)
        
        # Convert to embeddings
        # For binary classification, use [prob_down, prob_up] as 2D embedding
        # For higher dimensions, use feature importance or leaf indices
        if self.embedding_dim == 2:
            embeddings = np.column_stack([1 - xgb_pred, xgb_pred])
        else:
            # For higher dimensions, use leaf indices or feature contributions
            # This is a simplified approach - can be improved
            leaf_indices = self.xgb_model.predict(dtest, pred_leaf=True)
            # Convert leaf indices to embeddings (normalize to [-1, 1])
            embeddings = (leaf_indices.astype(float) / 100.0 - 0.5) * 2
            # Truncate/pad to embedding_dim
            if embeddings.shape[1] > self.embedding_dim:
                embeddings = embeddings[:, :self.embedding_dim]
            elif embeddings.shape[1] < self.embedding_dim:
                padding = np.zeros((embeddings.shape[0], self.embedding_dim - embeddings.shape[1]))
                embeddings = np.column_stack([embeddings, padding])
        
        # Normalize to [-1, 1] range (MANTIS requirement)
        embeddings = np.clip(embeddings, -1.0, 1.0)
        
        return embeddings
    
    def predict_binary(self, X_seq: np.ndarray) -> np.ndarray:
        """
        Predict binary direction (for evaluation)
        
        Args:
            X_seq: Input sequences
            
        Returns:
            Binary predictions (0 or 1)
        """
        embeddings = self.predict_embeddings(X_seq)
        
        # For 2D embeddings, use second dimension as probability
        if embeddings.shape[1] >= 2:
            probs = embeddings[:, 1]
        else:
            probs = (embeddings[:, 0] + 1) / 2  # Convert [-1, 1] to [0, 1]
        
        return (probs > 0.5).astype(int)
    
    def save(self, model_dir: str):
        """
        Save model to disk
        
        Args:
            model_dir: Directory to save model
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save LSTM
        if self.lstm_model is not None:
            self.lstm_model.save(os.path.join(model_dir, 'lstm_model.h5'))
        
        # Save XGBoost
        if self.xgb_model is not None:
            self.xgb_model.save_model(os.path.join(model_dir, 'xgb_model.json'))
        
        # Save scaler and feature indices
        with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(os.path.join(model_dir, 'feature_indices.pkl'), 'wb') as f:
            pickle.dump(self.selected_feature_indices, f)
        
        # Save config
        config = {
            'embedding_dim': self.embedding_dim,
            'lstm_hidden': self.lstm_hidden,
            'lstm_layers': self.lstm_layers,
            'time_steps': self.time_steps,
            'dropout': self.dropout
        }
        with open(os.path.join(model_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model saved to {model_dir}")
    
    def load(self, model_dir: str):
        """
        Load model from disk
        
        Args:
            model_dir: Directory containing saved model
        """
        if tf is None:
            print("TensorFlow not available, cannot load LSTM")
            return
        
        # Load config
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        # Load LSTM
        lstm_path = os.path.join(model_dir, 'lstm_model.h5')
        if os.path.exists(lstm_path):
            try:
                # Try loading with compile=False to avoid metric deserialization issues
                self.lstm_model = tf.keras.models.load_model(lstm_path, compile=False)
            except Exception as e:
                # If that fails, try loading normally
                try:
                    self.lstm_model = tf.keras.models.load_model(lstm_path)
                except Exception as e2:
                    # If both fail, try loading weights only
                    print(f"Warning: Could not load full model, trying weights only: {e2}")
                    # Reconstruct model architecture first
                    if hasattr(self, 'embedding_dim'):
                        # Model should already be initialized, just load weights
                        try:
                            self.lstm_model.load_weights(lstm_path.replace('.h5', '_weights.h5'))
                        except:
                            # Last resort: load from .h5 as weights
                            self.lstm_model.load_weights(lstm_path)
                    else:
                        raise e2
        
        # Load XGBoost
        xgb_path = os.path.join(model_dir, 'xgb_model.json')
        if os.path.exists(xgb_path):
            self.xgb_model = xgb.Booster()
            self.xgb_model.load_model(xgb_path)
        
        # Load scaler and feature indices
        with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(os.path.join(model_dir, 'feature_indices.pkl'), 'rb') as f:
            self.selected_feature_indices = pickle.load(f)
        
        print(f"Model loaded from {model_dir}")
