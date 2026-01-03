"""
Feature engineering module for trading signals.
Computes technical indicators, sentiment features, and macro features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator


class FeatureEngineer:
    """Engineering trading features from raw market data."""
    
    def __init__(self):
        pass
    
    def add_technical_indicators(
        self,
        df: pd.DataFrame,
        price_col: str = 'close'
    ) -> pd.DataFrame:
        """
        Add comprehensive technical indicators.
        
        Args:
            df: OHLCV DataFrame
            price_col: Column to use for price-based indicators
        
        Returns:
            DataFrame with added technical indicators
        """
        df = df.copy()
        
        # Trend indicators
        df['sma_20'] = SMAIndicator(close=df[price_col], window=20).sma_indicator()
        df['sma_50'] = SMAIndicator(close=df[price_col], window=50).sma_indicator()
        df['ema_12'] = EMAIndicator(close=df[price_col], window=12).ema_indicator()
        df['ema_26'] = EMAIndicator(close=df[price_col], window=26).ema_indicator()
        
        # MACD
        macd = MACD(close=df[price_col])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Momentum indicators
        df['rsi'] = RSIIndicator(close=df[price_col], window=14).rsi()
        
        stoch = StochasticOscillator(
            high=df['high'],
            low=df['low'],
            close=df[price_col]
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Volatility indicators
        bb = BollingerBands(close=df[price_col], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        
        df['atr'] = AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df[price_col]
        ).average_true_range()
        
        # Volume indicators
        df['obv'] = OnBalanceVolumeIndicator(
            close=df[price_col],
            volume=df['volume']
        ).on_balance_volume()
        
        # Price-based features
        df['returns'] = df[price_col].pct_change()
        df['log_returns'] = np.log(df[price_col] / df[price_col].shift(1))
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'returns_mean_{window}'] = df['returns'].rolling(window).mean()
            df[f'returns_std_{window}'] = df['returns'].rolling(window).std()
            df[f'volume_mean_{window}'] = df['volume'].rolling(window).mean()
        
        return df
    
    def add_sentiment_features(
        self,
        df: pd.DataFrame,
        news_df: pd.DataFrame,
        ticker: str
    ) -> pd.DataFrame:
        """
        Add sentiment features from news data.
        
        Args:
            df: Price DataFrame (must have datetime index)
            news_df: News DataFrame with columns [timestamp, ticker, sentiment]
            ticker: Ticker symbol to filter news
        
        Returns:
            DataFrame with sentiment features
        """
        df = df.copy()
        
        # Filter news for this ticker
        ticker_news = news_df[news_df['ticker'] == ticker].copy()
        ticker_news['date'] = pd.to_datetime(ticker_news['timestamp']).dt.date
        
        # Aggregate daily sentiment
        sentiment_mapping = {'positive': 1, 'neutral': 0, 'negative': -1}
        ticker_news['sentiment_score'] = ticker_news['sentiment'].map(sentiment_mapping)
        
        daily_sentiment = ticker_news.groupby('date').agg({
            'sentiment_score': ['mean', 'sum', 'count']
        })
        daily_sentiment.columns = ['sentiment_mean', 'sentiment_sum', 'news_count']
        
        # Merge with price data
        df['date'] = df.index.date
        df = df.merge(daily_sentiment, left_on='date', right_index=True, how='left')
        df = df.drop('date', axis=1)
        
        # Fill missing sentiment with neutral
        df['sentiment_mean'] = df['sentiment_mean'].fillna(0)
        df['sentiment_sum'] = df['sentiment_sum'].fillna(0)
        df['news_count'] = df['news_count'].fillna(0)
        
        # Rolling sentiment features
        for window in [3, 7, 14]:
            df[f'sentiment_mean_{window}d'] = df['sentiment_mean'].rolling(window).mean()
        
        return df
    
    def add_macro_features(
        self,
        df: pd.DataFrame,
        macro_data: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        Add macroeconomic features.
        
        Args:
            df: Price DataFrame
            macro_data: Dictionary of macro series
        
        Returns:
            DataFrame with macro features
        """
        df = df.copy()
        
        for series_name, series in macro_data.items():
            # Resample to daily and forward-fill
            series_daily = series.resample('D').ffill()
            
            # Merge with price data
            df = df.merge(
                series_daily.rename(series_name),
                left_index=True,
                right_index=True,
                how='left'
            )
            
            # Forward fill missing values
            df[series_name] = df[series_name].ffill().bfill()
            
            # Add change features
            df[f'{series_name}_change'] = df[series_name].pct_change()
            df[f'{series_name}_change_7d'] = df[series_name].pct_change(7)
        
        return df
    
    def create_feature_matrix(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],
        news_df: pd.DataFrame = None,
        macro_data: Dict[str, pd.Series] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Create complete feature matrix for all tickers.
        
        Args:
            ohlcv_data: Dictionary mapping ticker to OHLCV DataFrame
            news_df: Optional news DataFrame
            macro_data: Optional macro data dictionary
        
        Returns:
            Dictionary mapping ticker to feature DataFrame
        """
        feature_data = {}
        
        for ticker, df in ohlcv_data.items():
            # Start with technical indicators
            features = self.add_technical_indicators(df)
            
            # Add sentiment if available
            if news_df is not None:
                features = self.add_sentiment_features(features, news_df, ticker)
            
            # Add macro features if available
            if macro_data is not None:
                features = self.add_macro_features(features, macro_data)
            
            # Drop NaN rows from initial indicator calculation
            features = features.dropna()
            
            feature_data[ticker] = features
        
        return feature_data
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature column names (excluding OHLCV)."""
        base_cols = ['open', 'high', 'low', 'close', 'volume', 'adj_close']
        return [col for col in df.columns if col not in base_cols]


class TargetGenerator:
    """Generate trading targets (labels) from price data."""
    
    def __init__(self):
        pass
    
    def forward_returns(
        self,
        df: pd.DataFrame,
        horizons: List[int] = [1, 5, 10],
        price_col: str = 'close'
    ) -> pd.DataFrame:
        """
        Compute forward returns at multiple horizons.
        
        Args:
            df: Price DataFrame
            horizons: List of forward-looking periods
            price_col: Price column to use
        
        Returns:
            DataFrame with forward return columns
        """
        df = df.copy()
        
        for h in horizons:
            df[f'fwd_return_{h}d'] = df[price_col].pct_change(h).shift(-h)
        
        return df
    
    def direction_labels(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        threshold: float = 0.0,
        price_col: str = 'close'
    ) -> pd.DataFrame:
        """
        Generate directional labels (up/down).
        
        Args:
            df: Price DataFrame
            horizon: Forward-looking period
            threshold: Minimum return to classify as up/down
            price_col: Price column to use
        
        Returns:
            DataFrame with direction label column
        """
        df = df.copy()
        
        fwd_returns = df[price_col].pct_change(horizon).shift(-horizon)
        
        df[f'direction_{horizon}d'] = 0  # Neutral
        df.loc[fwd_returns > threshold, f'direction_{horizon}d'] = 1  # Up
        df.loc[fwd_returns < -threshold, f'direction_{horizon}d'] = -1  # Down
        
        return df
    
    def volatility_labels(
        self,
        df: pd.DataFrame,
        window: int = 20,
        price_col: str = 'close'
    ) -> pd.DataFrame:
        """
        Generate volatility regime labels.
        
        Args:
            df: Price DataFrame
            window: Lookback window for volatility
            price_col: Price column to use
        
        Returns:
            DataFrame with volatility label
        """
        df = df.copy()
        
        returns = df[price_col].pct_change()
        realized_vol = returns.rolling(window).std() * np.sqrt(252)
        
        # Classify into low/medium/high volatility
        vol_quantiles = realized_vol.quantile([0.33, 0.67])
        
        df['vol_regime'] = 1  # Medium
        df.loc[realized_vol < vol_quantiles[0.33], 'vol_regime'] = 0  # Low
        df.loc[realized_vol > vol_quantiles[0.67], 'vol_regime'] = 2  # High
        
        return df


if __name__ == "__main__":
    from market_data_loader import MarketDataLoader, SyntheticNewsGenerator
    from datetime import datetime, timedelta
    
    # Test feature engineering
    loader = MarketDataLoader()
    tickers = ['AAPL', 'MSFT']
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    
    # Fetch data
    ohlcv_data = loader.fetch_ohlcv(tickers, start_date, end_date)
    macro_data = loader.fetch_macro_data(['DFF', 'VIXCLS'], start_date, end_date)
    
    # Generate news
    news_gen = SyntheticNewsGenerator()
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    news_df = news_gen.generate_news(tickers, dates)
    
    # Engineer features
    engineer = FeatureEngineer()
    feature_data = engineer.create_feature_matrix(ohlcv_data, news_df, macro_data)
    
    # Print results
    for ticker, df in feature_data.items():
        print(f"\n{ticker}: {df.shape}")
        print(f"Features: {len(engineer.get_feature_names(df))}")
        print(df.tail())
    
    # Test target generation
    target_gen = TargetGenerator()
    labeled_data = target_gen.forward_returns(feature_data['AAPL'])
    labeled_data = target_gen.direction_labels(labeled_data)
    print(f"\nLabeled data shape: {labeled_data.shape}")
    print(labeled_data[['close', 'fwd_return_1d', 'direction_1d']].tail())
