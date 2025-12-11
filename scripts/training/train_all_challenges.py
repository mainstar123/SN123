"""
Train Models for All Challenges with Hyperparameter Tuning

This script:
1. Loads config.CHALLENGES
2. Trains a model for each challenge with hyperparameter tuning
3. Saves best model per challenge
4. Generates comprehensive training reports

Usage:
    python scripts/training/train_all_challenges.py [options]
    
Options:
    --quick: Skip hyperparameter tuning (faster, for testing)
    --challenge: Train only specific challenge (e.g., ETHUSDT-LBFGS-1H-17DIM)
    --trials: Number of Optuna trials (default: 50)
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
from datetime import datetime
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import config
from scripts.training.model_architecture import VMDTMFGLSTMXGBoost
from scripts.feature_engineering.feature_extractor import FeatureExtractor

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    tf = None

try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    print("⚠️  Optuna not installed. Install with: pip install optuna")
    HAS_OPTUNA = False


class ChallengeTrainer:
    """Train models for all challenges with hyperparameter tuning"""
    
    def __init__(
        self, 
        data_dir: str = "data", 
        output_dir: str = "models/tuned",
        use_tuning: bool = True,
        n_trials: int = 50
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.use_tuning = use_tuning and HAS_OPTUNA
        self.n_trials = n_trials
        self.challenges = config.CHALLENGES
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training log
        self.training_log = []
        
    def load_challenge_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load OHLCV data for a challenge"""
        # Try different file patterns
        patterns = [
            f"{ticker}_1h.csv",
            f"{ticker}_6h.csv",
            f"{ticker}.csv",
            f"{ticker.replace('USDT', '')}_1h.csv",
            f"{ticker.replace('-', '_')}.csv"
        ]
        
        for pattern in patterns:
            path = os.path.join(self.data_dir, pattern)
            if os.path.exists(path):
                print(f"  ✓ Loading data from {path}")
                df = pd.read_csv(path)
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                elif 'timestamp' in df.columns:
                    df['datetime'] = pd.to_datetime(df['timestamp'])
                    
                # Ensure required columns
                required = ['open', 'high', 'low', 'close', 'volume']
                if all(col in df.columns for col in required):
                    return df
                else:
                    print(f"  ⚠️  Missing required columns in {path}")
        
        print(f"  ✗ No data found for {ticker}")
        return None
    
    def split_data(self, df: pd.DataFrame, train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and validation sets"""
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()
        return train_df, val_df
    
    def create_objective(
        self, 
        challenge_name: str,
        embedding_dim: int,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame
    ):
        """Create Optuna objective function for hyperparameter tuning"""
        
        def objective(trial: optuna.Trial) -> float:
            """Optuna objective function"""
            
            # Hyperparameters to tune
            params = {
                'embedding_dim': embedding_dim,
                'lstm_hidden': trial.suggest_int('lstm_hidden', 64, 256, step=32),
                'lstm_layers': trial.suggest_int('lstm_layers', 1, 3),
                'time_steps': trial.suggest_int('time_steps', 10, 30, step=5),
                'vmd_k': trial.suggest_int('vmd_k', 5, 12),
                'tmfg_n_features': trial.suggest_int('tmfg_n_features', 8, 15),
                'dropout': trial.suggest_float('dropout', 0.1, 0.4),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
                'use_gpu': True
            }
            
            try:
                # Create model with trial's hyperparameters
                model = VMDTMFGLSTMXGBoost(**params)
                
                # Prepare data with this trial's feature extraction settings
                # This ensures tmfg_n_features matches between model and data
                X_train, y_train, _ = model.prepare_data(train_df)
                X_val, y_val, _ = model.prepare_data(val_df)
                
                if len(X_train) == 0 or len(X_val) == 0:
                    return float('inf')
                
                # Train model
                model.train(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    epochs=50,  # Reduced for tuning
                    batch_size=64,
                    verbose=0
                )
                
                # Evaluate on validation set
                val_embeddings = model.predict_embeddings(X_val)
                
                # Calculate validation loss (MSE between embeddings and targets)
                # For binary challenges, we want embeddings that separate well
                if embedding_dim == 2:
                    # For 2D, use XGBoost accuracy as metric
                    val_pred_binary = model.predict_binary(X_val)
                    threshold = np.percentile(y_val, 50)
                    y_val_binary = (y_val > threshold).astype(int)
                    accuracy = np.mean(val_pred_binary == y_val_binary)
                    # Maximize accuracy (minimize negative accuracy)
                    return -accuracy
                else:
                    # For higher dimensions, use embedding quality
                    # MSE between consecutive embeddings (smoothness)
                    embedding_diff = np.diff(val_embeddings, axis=0)
                    smoothness = np.mean(np.linalg.norm(embedding_diff, axis=1))
                    return smoothness
                    
            except Exception as e:
                print(f"    Trial failed: {str(e)[:100]}")
                return float('inf')
        
        return objective
    
    def train_challenge(
        self, 
        challenge_name: str,
        embedding_dim: int,
        challenge_weight: float,
        ticker: str = None,
        price_key: str = None
    ) -> Dict:
        """Train model for a single challenge"""
        
        print(f"\n{'='*80}")
        print(f"🎯 Training Challenge: {challenge_name}")
        print(f"   Embedding Dim: {embedding_dim}, Weight: {challenge_weight}")
        print(f"{'='*80}")
        
        start_time = datetime.now()
        
        # Use price_key if available (for challenges like ETHLBFGS that use ETH data)
        # Otherwise use the provided ticker, or extract from challenge name
        if price_key:
            data_ticker = price_key
            print(f"  Using price_key: {price_key}")
        elif ticker:
            data_ticker = ticker
        else:
            data_ticker = challenge_name.split('-')[0]
        
        # Load data
        df = self.load_challenge_data(data_ticker)
        if df is None or len(df) < 100:
            print(f"  ✗ Insufficient data for {challenge_name}")
            return {
                'challenge_name': challenge_name,
                'status': 'failed',
                'reason': 'insufficient_data'
            }
        
        print(f"  Data loaded: {len(df)} rows")
        
        # Split data
        train_df, val_df = self.split_data(df, train_ratio=0.7)
        print(f"  Train: {len(train_df)} rows, Val: {len(val_df)} rows")
        
        try:
            # Default hyperparameters
            best_params = {
                'embedding_dim': embedding_dim,
                'lstm_hidden': 128,
                'lstm_layers': 2,
                'time_steps': 20,
                'vmd_k': 8,
                'tmfg_n_features': 10,
                'dropout': 0.2,
                'learning_rate': 0.0005,
                'use_gpu': True
            }
            
            # Hyperparameter tuning with Optuna
            if self.use_tuning:
                print(f"\n  🔧 Starting hyperparameter tuning ({self.n_trials} trials)...")
                if HAS_TF and tf.config.list_physical_devices('GPU'):
                    print(f"  Model will use GPU: {tf.config.list_physical_devices('GPU')[0].name}")
                else:
                    print("  Model will use CPU")
                
                # Verify we have enough data for at least one sequence
                # Use default time_steps=20 for the check
                if len(train_df) < 20 or len(val_df) < 20:
                    print(f"  ✗ Insufficient data for sequences (need at least 20 rows)")
                    return {
                        'challenge_name': challenge_name,
                        'status': 'failed',
                        'reason': 'insufficient_data'
                    }
                
                # Create Optuna study
                study = optuna.create_study(
                    direction='minimize',
                    sampler=TPESampler(seed=42)
                )
                
                # Run optimization - pass raw dataframes so each trial can prepare data with its own settings
                objective = self.create_objective(
                    challenge_name, embedding_dim,
                    train_df, val_df
                )
                
                study.optimize(
                    objective,
                    n_trials=self.n_trials,
                    show_progress_bar=True,
                    callbacks=[
                        lambda study, trial: print(f"    Trial {trial.number}: {trial.value:.6f}")
                    ]
                )
                
                # Get best parameters
                best_params.update(study.best_params)
                print(f"\n  ✓ Best parameters found:")
                for k, v in study.best_params.items():
                    print(f"    {k}: {v}")
                print(f"  Best score: {study.best_value:.6f}")
            
            # Train final model with best parameters
            print(f"\n  🚀 Training final model with best parameters...")
            
            model = VMDTMFGLSTMXGBoost(**best_params)
            
            # Prepare data
            X_train, y_train, feature_names = model.prepare_data(train_df)
            X_val, y_val, _ = model.prepare_data(val_df)
            
            print(f"  Final sequences: Train={len(X_train)}, Val={len(X_val)}")
            print(f"  Features selected: {len(feature_names)}")
            
            # Train
            model.train(
                X_train,
                y_train,
                X_val,
                y_val,
                epochs=100,
                batch_size=64,
                verbose=1
            )
            
            # Evaluate
            train_pred = model.predict_binary(X_train)
            val_pred = model.predict_binary(X_val)
            
            threshold = np.percentile(y_train, 50)
            y_train_binary = (y_train > threshold).astype(int)
            y_val_binary = (y_val > threshold).astype(int)
            
            train_acc = np.mean(train_pred == y_train_binary)
            val_acc = np.mean(val_pred == y_val_binary)
            
            print(f"\n  📊 Final Results:")
            print(f"    Train Accuracy: {train_acc:.4f}")
            print(f"    Val Accuracy: {val_acc:.4f}")
            
            # Save model
            model_dir = os.path.join(self.output_dir, challenge_name)
            model.save(model_dir)
            
            # Save hyperparameters
            params_file = os.path.join(model_dir, 'best_params.json')
            with open(params_file, 'w') as f:
                json.dump(best_params, f, indent=2)
            
            # Save feature names
            features_file = os.path.join(model_dir, 'features.txt')
            with open(features_file, 'w') as f:
                for fname in feature_names:
                    f.write(f"{fname}\n")
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            result = {
                'challenge_name': challenge_name,
                'status': 'success',
                'embedding_dim': embedding_dim,
                'weight': challenge_weight,
                'train_accuracy': float(train_acc),
                'val_accuracy': float(val_acc),
                'best_params': best_params,
                'duration_seconds': duration,
                'model_dir': model_dir
            }
            
            print(f"\n  ✓ Training completed in {duration:.1f}s")
            print(f"  Model saved to: {model_dir}")
            
            return result
            
        except Exception as e:
            print(f"\n  ✗ Training failed: {str(e)}")
            print(f"  Traceback: {traceback.format_exc()}")
            return {
                'challenge_name': challenge_name,
                'status': 'failed',
                'reason': str(e),
                'traceback': traceback.format_exc()
            }
    
    def train_all_challenges(self, specific_challenge: Optional[str] = None):
        """Train models for all challenges"""
        
        print("\n" + "="*80)
        print("🚀 MANTIS Multi-Challenge Training Pipeline")
        print("="*80)
        print(f"Data Directory: {self.data_dir}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Hyperparameter Tuning: {'Enabled' if self.use_tuning else 'Disabled'}")
        if self.use_tuning:
            print(f"Trials per Challenge: {self.n_trials}")
        print("="*80)
        
        # Filter challenges if specific one requested
        if specific_challenge:
            challenges_to_train = [
                ch for ch in self.challenges 
                if ch['name'] == specific_challenge
            ]
            if not challenges_to_train:
                print(f"✗ Challenge '{specific_challenge}' not found in config.CHALLENGES")
                return
        else:
            challenges_to_train = self.challenges
        
        print(f"\n📋 Total Challenges to Train: {len(challenges_to_train)}")
        for ch in challenges_to_train:
            print(f"  - {ch['name']} (dim={ch['dim']}, weight={ch['weight']})")
        
        # Sort by weight (train high-weight challenges first)
        challenges_sorted = sorted(
            challenges_to_train,
            key=lambda x: x['weight'],
            reverse=True
        )
        
        print(f"\n🎯 Training Order (by weight):")
        for i, ch in enumerate(challenges_sorted, 1):
            print(f"  {i}. {ch['name']} (weight={ch['weight']})")
        
        # Train each challenge
        results = []
        successful = 0
        failed = 0
        
        for i, challenge in enumerate(challenges_sorted, 1):
            print(f"\n\n{'#'*80}")
            print(f"Progress: {i}/{len(challenges_sorted)}")
            print(f"{'#'*80}")
            
            result = self.train_challenge(
                challenge['name'],
                challenge['dim'],
                challenge['weight'],
                ticker=challenge.get('ticker'),
                price_key=challenge.get('price_key')
            )
            
            results.append(result)
            self.training_log.append(result)
            
            if result['status'] == 'success':
                successful += 1
            else:
                failed += 1
            
            # Save intermediate results
            self.save_results(results)
        
        # Final summary
        print("\n\n" + "="*80)
        print("🎉 TRAINING COMPLETE!")
        print("="*80)
        print(f"Total Challenges: {len(challenges_sorted)}")
        print(f"✓ Successful: {successful}")
        print(f"✗ Failed: {failed}")
        print("="*80)
        
        # Print detailed results
        print("\n📊 Detailed Results:")
        for result in results:
            if result['status'] == 'success':
                print(f"\n✓ {result['challenge_name']}:")
                print(f"  Train Acc: {result['train_accuracy']:.4f}")
                print(f"  Val Acc: {result['val_accuracy']:.4f}")
                print(f"  Duration: {result['duration_seconds']:.1f}s")
            else:
                print(f"\n✗ {result['challenge_name']}: {result['reason']}")
        
        print("\n" + "="*80)
        print(f"Results saved to: {self.output_dir}/training_results.json")
        print("="*80)
        
        return results
    
    def save_results(self, results: List[Dict]):
        """Save training results to JSON"""
        results_file = os.path.join(self.output_dir, 'training_results.json')
        
        # Add timestamp
        output = {
            'timestamp': datetime.now().isoformat(),
            'total_challenges': len(results),
            'successful': sum(1 for r in results if r['status'] == 'success'),
            'failed': sum(1 for r in results if r['status'] == 'failed'),
            'use_tuning': self.use_tuning,
            'n_trials': self.n_trials if self.use_tuning else 0,
            'results': results
        }
        
        with open(results_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='Train models for all MANTIS challenges',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Data directory (default: data)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='models/tuned',
        help='Output directory for models (default: models/tuned)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Skip hyperparameter tuning (faster, for testing)'
    )
    
    parser.add_argument(
        '--challenge',
        type=str,
        help='Train only specific challenge (e.g., ETH-LBFGS)'
    )
    
    parser.add_argument(
        '--trials',
        type=int,
        default=50,
        help='Number of Optuna trials for hyperparameter tuning (default: 50)'
    )
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = ChallengeTrainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        use_tuning=not args.quick,
        n_trials=args.trials
    )
    
    # Train all challenges
    results = trainer.train_all_challenges(specific_challenge=args.challenge)
    
    print("\n✅ All done!")


if __name__ == '__main__':
    main()

