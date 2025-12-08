"""
Feature Engineering Module
Implements VMD, TMFG, and orthogonal signal generation for MANTIS
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from scipy import signal
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    from vmdpy import VMD
except ImportError:
    print("Installing vmdpy...")
    import os
    os.system("pip install vmdpy")
    from vmdpy import VMD


class FeatureEngineer:
    """Feature engineering for MANTIS models"""
    
    def __init__(self, n_vmd_modes: int = 8, n_tmfg_features: int = 10):
        """
        Initialize feature engineer
        
        Args:
            n_vmd_modes: Number of VMD modes (default: 8)
            n_tmfg_features: Number of TMFG-selected features (default: 10)
        """
        self.n_vmd_modes = n_vmd_modes
        self.n_tmfg_features = n_tmfg_features
        self.scaler = StandardScaler()
        self.selected_feature_indices = None
        
    def vmd_decompose(
        self,
        prices: np.ndarray,
        alpha: float = 2000.0,
        tau: float = 0.0,
        K: int = None,
        DC: int = 0,
        init: int = 1,
        tol: float = 1e-7
    ) -> np.ndarray:
        """
        Variational Mode Decomposition
        
        Args:
            prices: Price series (1D array)
            alpha: Balancing parameter
            tau: Noise-tolerance
            K: Number of modes (default: self.n_vmd_modes)
            DC: DC component flag
            init: Initialization method
            tol: Tolerance
            
        Returns:
            Array of shape (K, len(prices)) with IMF components
        """
        if K is None:
            K = self.n_vmd_modes
        
        try:
            u, u_hat, omega = VMD(prices, alpha, tau, K, DC, init, tol)
            return u
        except Exception as e:
            print(f"VMD error: {e}, using fallback")
            # Fallback: simple moving average decomposition
            return self._fallback_decomposition(prices, K)
    
    def _fallback_decomposition(self, prices: np.ndarray, K: int) -> np.ndarray:
        """Fallback decomposition using moving averages"""
        modes = []
        for k in range(K):
            window = 2 ** (k + 1)
            if window > len(prices):
                window = len(prices)
            ma = pd.Series(prices).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
            modes.append(ma.values)
        return np.array(modes)
    
    def compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical indicators
        
        Args:
            df: DataFrame with OHLCV data
            
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
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price patterns
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
        
        # Volatility
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
        
        return df
    
    def compute_oi_funding_features(
        self,
        df: pd.DataFrame,
        funding_df: Optional[pd.DataFrame] = None,
        oi_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Compute OI and funding rate features (orthogonal signals)
        
        Args:
            df: Main price DataFrame
            funding_df: DataFrame with funding rates
            oi_df: DataFrame with open interest
            
        Returns:
            DataFrame with additional OI/funding features
        """
        df = df.copy()
        
        # Merge funding rates
        if funding_df is not None and not funding_df.empty:
            funding_df = funding_df.copy()
            funding_df['datetime'] = pd.to_datetime(funding_df['datetime'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Merge on nearest hour
            df = df.merge(
                funding_df[['datetime', 'funding_rate']],
                on='datetime',
                how='left'
            )
            df['funding_rate'] = df['funding_rate'].fillna(method='ffill').fillna(0)
            
            # Funding rate features
            df['funding_rate_ma'] = df['funding_rate'].rolling(window=8).mean()  # 8h MA
            df['funding_rate_deviation'] = df['funding_rate'] - df['funding_rate_ma']
            df['funding_rate_momentum'] = df['funding_rate'].diff()
        else:
            df['funding_rate'] = 0.0
            df['funding_rate_ma'] = 0.0
            df['funding_rate_deviation'] = 0.0
            df['funding_rate_momentum'] = 0.0
        
        # Merge open interest
        if oi_df is not None and not oi_df.empty:
            oi_df = oi_df.copy()
            oi_df['datetime'] = pd.to_datetime(oi_df['datetime'])
            
            df = df.merge(
                oi_df[['datetime', 'open_interest']],
                on='datetime',
                how='left'
            )
            df['open_interest'] = df['open_interest'].fillna(method='ffill').fillna(0)
            
            # OI features
            df['oi_delta'] = df['open_interest'].diff()
            df['oi_delta_pct'] = df['open_interest'].pct_change()
            df['oi_ma'] = df['open_interest'].rolling(window=24).mean()  # 24h MA
            df['oi_ratio'] = df['open_interest'] / df['oi_ma']
        else:
            df['open_interest'] = 0.0
            df['oi_delta'] = 0.0
            df['oi_delta_pct'] = 0.0
            df['oi_ma'] = 0.0
            df['oi_ratio'] = 0.0
        
        # Orthogonal signal: OI/Funding interaction
        df['oi_funding_interaction'] = (
            df['funding_rate_deviation'] * df['oi_delta_pct']
        )
        
        return df
    
    def compute_cross_exchange_features(
        self,
        df_binance: pd.DataFrame,
        df_bybit: pd.DataFrame,
        funding_binance: Optional[pd.DataFrame] = None,
        funding_bybit: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Compute cross-exchange features (arbitrage signals)
        
        Args:
            df_binance: Binance price data
            df_bybit: Bybit price data
            funding_binance: Binance funding rates
            funding_bybit: Bybit funding rates
            
        Returns:
            DataFrame with cross-exchange features
        """
        # Merge on datetime
        df_binance = df_binance.copy()
        df_bybit = df_bybit.copy()
        df_binance['datetime'] = pd.to_datetime(df_binance['datetime'])
        df_bybit['datetime'] = pd.to_datetime(df_bybit['datetime'])
        
        # Price divergence
        merged = df_binance[['datetime', 'close']].merge(
            df_bybit[['datetime', 'close']],
            on='datetime',
            suffixes=('_binance', '_bybit'),
            how='inner'
        )
        merged['price_divergence'] = (
            merged['close_binance'] - merged['close_bybit']
        ) / merged['close_binance']
        
        # Funding rate divergence
        if funding_binance is not None and funding_bybit is not None:
            funding_binance = funding_binance.copy()
            funding_bybit = funding_bybit.copy()
            funding_binance['datetime'] = pd.to_datetime(funding_binance['datetime'])
            funding_bybit['datetime'] = pd.to_datetime(funding_bybit['datetime'])
            
            merged = merged.merge(
                funding_binance[['datetime', 'funding_rate']],
                on='datetime',
                how='left'
            )
            merged = merged.rename(columns={'funding_rate': 'funding_binance'})
            
            merged = merged.merge(
                funding_bybit[['datetime', 'funding_rate']],
                on='datetime',
                how='left'
            )
            merged = merged.rename(columns={'funding_rate': 'funding_bybit'})
            
            merged['funding_divergence'] = (
                merged['funding_binance'] - merged['funding_bybit']
            )
        else:
            merged['funding_divergence'] = 0.0
        
        return merged[['datetime', 'price_divergence', 'funding_divergence']]
    
    def tmfg_feature_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_features: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        TMFG feature selection (approximated via Random Forest importance)
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target variable
            n_features: Number of features to select (default: self.n_tmfg_features)
            
        Returns:
            Selected features and indices
        """
        if n_features is None:
            n_features = self.n_tmfg_features
        
        # Remove NaN/inf
        mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1) | np.isnan(y) | np.isinf(y))
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) < 100:
            # Not enough data, return all features
            return X, np.arange(X.shape[1])
        
        # Use Random Forest for feature importance (TMFG approximation)
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_clean, y_clean)
        
        # Get feature importances
        importances = rf.feature_importances_
        top_indices = np.argsort(importances)[-n_features:]
        
        self.selected_feature_indices = top_indices
        
        return X[:, top_indices], top_indices
    
    def create_feature_matrix(
        self,
        df: pd.DataFrame,
        vmd_modes: Optional[np.ndarray] = None,
        include_technical: bool = True,
        include_oi_funding: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Create complete feature matrix
        
        Args:
            df: DataFrame with all features
            vmd_modes: VMD decomposition modes (optional)
            include_technical: Include technical indicators
            include_oi_funding: Include OI/funding features
            
        Returns:
            Feature matrix and feature names
        """
        feature_cols = []
        
        # Raw price features
        raw_cols = ['open', 'high', 'low', 'close']
        feature_cols.extend(raw_cols)
        
        # VMD modes
        if vmd_modes is not None:
            for i in range(vmd_modes.shape[0]):
                df[f'vmd_mode_{i}'] = vmd_modes[i]
                feature_cols.append(f'vmd_mode_{i}')
        
        # Technical indicators
        if include_technical:
            tech_cols = [
                'returns', 'sma_20', 'sma_50', 'rsi', 'macd', 'macd_signal',
                'bb_width', 'bb_position', 'atr', 'volume_ratio',
                'momentum_10', 'volatility_10'
            ]
            feature_cols.extend([c for c in tech_cols if c in df.columns])
        
        # OI/Funding features
        if include_oi_funding:
            oi_funding_cols = [
                'funding_rate', 'funding_rate_deviation', 'funding_rate_momentum',
                'oi_delta_pct', 'oi_ratio', 'oi_funding_interaction'
            ]
            feature_cols.extend([c for c in oi_funding_cols if c in df.columns])
        
        # Extract features
        X = df[feature_cols].values
        
        # Handle NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X, feature_cols
    
    def prepare_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timesteps: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM
        
        Args:
            X: Feature matrix
            y: Target variable
            timesteps: Number of timesteps
            
        Returns:
            X_seq, y_seq
        """
        X_seq, y_seq = [], []
        
        for i in range(timesteps, len(X)):
            X_seq.append(X[i-timesteps:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)


if __name__ == "__main__":
    # Test feature engineering
    print("Testing Feature Engineering...")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=1000, freq='1h')
    prices = 50000 + np.cumsum(np.random.randn(1000) * 100)
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': prices + np.random.randn(1000) * 10,
        'high': prices + np.abs(np.random.randn(1000) * 20),
        'low': prices - np.abs(np.random.randn(1000) * 20),
        'close': prices,
        'volume': np.random.rand(1000) * 1000000
    })
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # VMD decomposition
    print("Testing VMD...")
    vmd_modes = fe.vmd_decompose(df['close'].values)
    print(f"VMD modes shape: {vmd_modes.shape}")
    
    # Technical indicators
    print("Computing technical indicators...")
    df = fe.compute_technical_indicators(df)
    print(f"DataFrame shape: {df.shape}")
    
    # Feature matrix
    print("Creating feature matrix...")
    X, feature_names = fe.create_feature_matrix(df, vmd_modes=vmd_modes)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    # TMFG selection
    print("Testing TMFG feature selection...")
    y = df['close'].values[1:] - df['close'].values[:-1]
    X_selected, indices = fe.tmfg_feature_selection(X[:-1], y)
    print(f"Selected features shape: {X_selected.shape}")
    print(f"Selected indices: {indices}")
    
    print("\n✓ Feature engineering test complete!")

