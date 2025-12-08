"""
Feature Engineering Module for MANTIS Mining
Implements VMD (Variational Mode Decomposition), TMFG approximation, and technical indicators
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    from vmdpy import VMD
except ImportError:
    print("Warning: vmdpy not installed. Install with: pip install vmdpy")
    VMD = None


class FeatureExtractor:
    """Feature extraction with VMD, TMFG, and technical indicators"""
    
    def __init__(self, vmd_k: int = 8, vmd_alpha: float = 2000.0, 
                 tmfg_n_features: int = 10):
        """
        Initialize feature extractor
        
        Args:
            vmd_k: Number of VMD components (IMFs)
            vmd_alpha: VMD bandwidth constraint
            tmfg_n_features: Number of features to select via TMFG approximation
        """
        self.vmd_k = vmd_k
        self.vmd_alpha = vmd_alpha
        self.tmfg_n_features = tmfg_n_features
        self.selected_feature_indices = None
        self.scaler = StandardScaler()
        
    def vmd_decompose(self, prices: np.ndarray) -> np.ndarray:
        """
        Variational Mode Decomposition
        
        Args:
            prices: 1D array of prices
            
        Returns:
            Array of shape (vmd_k, len(prices)) with IMF components
        """
        if VMD is None:
            # Fallback: return original prices repeated
            print("Warning: VMD not available, using original prices")
            return np.tile(prices.reshape(1, -1), (self.vmd_k, 1))
        
        try:
            # VMD parameters
            tau = 0.0  # noise-tolerance
            DC = 0  # no DC part
            init = 1  # initialize omegas uniformly
            tol = 1e-7
            
            u, u_hat, omega = VMD(prices, self.vmd_alpha, tau, self.vmd_k, 
                                  DC, init, tol)
            
            return u  # Shape: (K, N)
            
        except Exception as e:
            print(f"VMD decomposition error: {e}, using fallback")
            return np.tile(prices.reshape(1, -1), (self.vmd_k, 1))
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with additional technical indicator columns
        """
        df = df.copy()
        
        # Price-based indicators
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        
        # Price position relative to MA
        for window in [20, 50, 200]:
            df[f'price_sma_{window}_ratio'] = df['close'] / df[f'sma_{window}']
            df[f'price_ema_{window}_ratio'] = df['close'] / df[f'ema_{window}']
        
        # Volatility indicators
        for window in [10, 20, 50]:
            df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
            df[f'atr_{window}'] = self._calculate_atr(df, window)
        
        # RSI
        for window in [14, 21]:
            df[f'rsi_{window}'] = self._calculate_rsi(df['close'], window)
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        for window in [20, 50]:
            sma = df['close'].rolling(window=window).mean()
            std = df['close'].rolling(window=window).std()
            df[f'bb_upper_{window}'] = sma + 2 * std
            df[f'bb_lower_{window}'] = sma - 2 * std
            df[f'bb_width_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / sma
            df[f'bb_position_{window}'] = (df['close'] - df[f'bb_lower_{window}']) / (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'])
        
        # Volume indicators
        if 'volume' in df.columns:
            df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            df['price_volume'] = df['close'] * df['volume']
            
            # On-Balance Volume (OBV)
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # High-Low spread
        df['hl_spread'] = (df['high'] - df['low']) / df['close']
        df['oc_spread'] = (df['close'] - df['open']) / df['open']
        
        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = df['close'].pct_change(period)
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    def _calculate_rsi(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def add_funding_features(self, df: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add funding rate features
        
        Args:
            df: OHLCV DataFrame
            funding_df: Funding rates DataFrame
            
        Returns:
            DataFrame with funding features added
        """
        if funding_df.empty:
            return df
        
        df = df.copy()
        
        # Merge funding rates (forward fill to align with hourly candles)
        df['datetime'] = pd.to_datetime(df['datetime'])
        funding_df['datetime'] = pd.to_datetime(funding_df['datetime'])
        
        # Resample funding rates to hourly and forward fill
        funding_hourly = funding_df.set_index('datetime').resample('1H').last().ffill()
        df = df.set_index('datetime').join(funding_hourly[['funding_rate']], how='left')
        df['funding_rate'] = df['funding_rate'].ffill()
        df = df.reset_index()
        
        # Funding rate features
        if 'funding_rate' in df.columns:
            df['funding_rate_ma_8'] = df['funding_rate'].rolling(window=8).mean()
            df['funding_rate_ma_24'] = df['funding_rate'].rolling(window=24).mean()
            df['funding_rate_deviation'] = df['funding_rate'] - df['funding_rate_ma_24']
            df['funding_rate_momentum'] = df['funding_rate'].diff(8)
        
        return df
    
    def add_oi_features(self, df: pd.DataFrame, oi_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add open interest features (placeholder - would need OI data)
        
        Args:
            df: OHLCV DataFrame
            oi_df: Open interest DataFrame
            
        Returns:
            DataFrame with OI features added
        """
        if oi_df.empty:
            # Create dummy OI features based on volume
            df['oi_delta'] = df['volume'].diff() if 'volume' in df.columns else 0
            df['oi_ma'] = df['volume'].rolling(window=24).mean() if 'volume' in df.columns else 0
            df['oi_delta_ratio'] = df['oi_delta'] / (df['oi_ma'] + 1e-8)
            return df
        
        # Similar to funding features, merge OI data
        df = df.copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        oi_df['datetime'] = pd.to_datetime(oi_df['datetime'])
        
        oi_hourly = oi_df.set_index('datetime').resample('1H').last().ffill()
        df = df.set_index('datetime').join(oi_hourly, how='left')
        df = df.reset_index()
        
        # OI features
        if 'open_interest' in df.columns:
            df['oi_delta'] = df['open_interest'].diff()
            df['oi_ma'] = df['open_interest'].rolling(window=24).mean()
            df['oi_delta_ratio'] = df['oi_delta'] / (df['oi_ma'] + 1e-8)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features (OI/funding, cross-exchange signals, etc.)
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        # OI/Funding interaction (if both available)
        if 'funding_rate' in df.columns and 'oi_delta_ratio' in df.columns:
            df['oi_funding_interaction'] = (
                df['funding_rate_deviation'] * df['oi_delta_ratio']
            )
        
        # Volume profile deviation
        if 'volume' in df.columns:
            df['volume_percentile'] = df['volume'].rolling(window=168).rank(pct=True)  # 1 week
            df['volume_deviation'] = df['volume_percentile'] - 0.5
        
        # Price-volume divergence
        if 'volume' in df.columns:
            price_change = df['close'].pct_change()
            volume_change = df['volume'].pct_change()
            df['price_volume_divergence'] = price_change - volume_change
        
        return df
    
    def extract_all_features(self, df: pd.DataFrame, funding_df: pd.DataFrame = None,
                            oi_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Extract all features (technical indicators + VMD + funding + OI)
        
        Args:
            df: OHLCV DataFrame
            funding_df: Funding rates DataFrame (optional)
            oi_df: Open interest DataFrame (optional)
            
        Returns:
            DataFrame with all features
        """
        # Technical indicators
        df_features = self.calculate_technical_indicators(df)
        
        # Get dataframe length and index
        df_length = len(df_features)
        df_index = df_features.index
        
        # VMD decomposition - use close prices, ensuring same length
        prices = df_features['close'].values
        if len(prices) != df_length:
            # Fallback to original df if needed
            prices = df['close'].values[:df_length]
        
        # Perform VMD decomposition
        vmd_components = self.vmd_decompose(prices)  # Shape: (K, N)
        
        # Ensure VMD components match dataframe length exactly
        vmd_length = vmd_components.shape[1]
        for i in range(self.vmd_k):
            component = vmd_components[i]
            
            # Adjust length to match dataframe
            if len(component) != df_length:
                if len(component) < df_length:
                    # Pad with last value (forward fill)
                    last_val = component[-1] if len(component) > 0 else 0.0
                    padding = np.full(df_length - len(component), last_val)
                    component = np.concatenate([component, padding])
                else:
                    # Truncate to match
                    component = component[:df_length]
            
            # Create Series with proper index
            df_features[f'vmd_imf_{i+1}'] = pd.Series(component, index=df_index)
        
        # Funding features
        if funding_df is not None:
            df_features = self.add_funding_features(df_features, funding_df)
        
        # OI features
        if oi_df is not None:
            df_features = self.add_oi_features(df_features, oi_df)
        
        # Interaction features
        df_features = self.create_interaction_features(df_features)
        
        return df_features
    
    def tmfg_feature_selection(self, X: np.ndarray, y: np.ndarray, 
                               n_features: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        TMFG feature selection approximation using Random Forest importance
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            n_features: Number of features to select (default: self.tmfg_n_features)
            
        Returns:
            Tuple of (selected_features, selected_indices)
        """
        if n_features is None:
            n_features = self.tmfg_n_features
        
        # Remove NaN/inf
        mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1) | np.isnan(y) | np.isinf(y))
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) < 100:
            # Not enough data, return all features
            return X, np.arange(X.shape[1])
        
        # Fill remaining NaN with 0
        X_clean = np.nan_to_num(X_clean, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Use Random Forest to approximate TMFG feature importance
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        try:
            rf.fit(X_clean, y_clean)
            importances = rf.feature_importances_
            
            # Select top N features
            top_indices = np.argsort(importances)[-n_features:]
            selected_X = X[:, top_indices]
            
            self.selected_feature_indices = top_indices
            
            return selected_X, top_indices
            
        except Exception as e:
            print(f"Feature selection error: {e}, using all features")
            return X, np.arange(X.shape[1])
    
    def prepare_feature_matrix(self, df: pd.DataFrame, 
                              target_col: str = 'close') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare feature matrix from DataFrame
        
        Args:
            df: DataFrame with features
            target_col: Target column name
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        # Select numeric columns (exclude datetime and target)
        exclude_cols = ['datetime', target_col, 'open_time', 'close_time']
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and df[col].dtype in [np.float64, np.int64, np.float32, np.int32]]
        
        # Extract features and target
        X = df[feature_cols].values
        y = df[target_col].values if target_col in df.columns else None
        
        # Handle NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if y is not None:
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X, y, feature_cols


