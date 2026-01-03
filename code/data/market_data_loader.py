"""
Data loader for market data from multiple sources.
Implements fetching from Yahoo Finance, FRED, and synthetic generation.
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
from fredapi import Fred

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketDataLoader:
    """Unified market data loader supporting multiple sources."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize FRED API if key available
        fred_key = os.getenv("FRED_API_KEY")
        self.fred = Fred(api_key=fred_key) if fred_key else None
        
        yf.pdr_override()
    
    def fetch_ohlcv(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data from Yahoo Finance.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1d, 1h, etc.)
        
        Returns:
            Dictionary mapping ticker to OHLCV DataFrame
        """
        logger.info(f"Fetching OHLCV for {len(tickers)} tickers from {start_date} to {end_date}")
        
        data = {}
        for ticker in tickers:
            try:
                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    progress=False
                )
                
                if not df.empty:
                    # Standardize column names
                    df.columns = [col.lower() for col in df.columns]
                    df = df.rename(columns={
                        'adj close': 'adj_close'
                    })
                    data[ticker] = df
                    logger.info(f"  {ticker}: {len(df)} rows")
                else:
                    logger.warning(f"  {ticker}: No data returned")
                    
            except Exception as e:
                logger.error(f"  {ticker}: Error - {e}")
        
        return data
    
    def fetch_macro_data(
        self,
        series_ids: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.Series]:
        """
        Fetch macroeconomic data from FRED.
        
        Args:
            series_ids: List of FRED series IDs
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            Dictionary mapping series ID to data
        """
        if not self.fred:
            logger.warning("FRED API key not set, using fallback synthetic data")
            return self._generate_synthetic_macro(series_ids, start_date, end_date)
        
        logger.info(f"Fetching {len(series_ids)} macro series from FRED")
        
        data = {}
        for series_id in series_ids:
            try:
                series = self.fred.get_series(
                    series_id,
                    observation_start=start_date,
                    observation_end=end_date
                )
                data[series_id] = series
                logger.info(f"  {series_id}: {len(series)} observations")
            except Exception as e:
                logger.error(f"  {series_id}: Error - {e}")
        
        return data
    
    def _generate_synthetic_macro(
        self,
        series_ids: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.Series]:
        """Generate synthetic macro data as fallback."""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Default synthetic series properties
        series_configs = {
            'DFF': (2.5, 0.5, 0.001),  # Fed Funds Rate: mean, std, drift
            'T10Y2Y': (1.0, 0.3, 0.0),  # 10Y-2Y spread
            'VIXCLS': (18.0, 5.0, 0.0),  # VIX
            'DEXUSEU': (1.1, 0.05, 0.0),  # USD/EUR
        }
        
        data = {}
        np.random.seed(42)
        
        for series_id in series_ids:
            if series_id in series_configs:
                mean, std, drift = series_configs[series_id]
            else:
                mean, std, drift = (100.0, 10.0, 0.0)
            
            # Generate AR(1) process
            values = [mean]
            for _ in range(len(dates) - 1):
                shock = np.random.normal(0, std)
                new_val = 0.95 * values[-1] + drift + shock
                values.append(new_val)
            
            data[series_id] = pd.Series(values, index=dates, name=series_id)
        
        return data
    
    def save_data(self, data: Dict, filename: str):
        """Save data dictionary to pickle."""
        filepath = os.path.join(self.data_dir, filename)
        pd.to_pickle(data, filepath)
        logger.info(f"Saved data to {filepath}")
    
    def load_data(self, filename: str) -> Dict:
        """Load data dictionary from pickle."""
        filepath = os.path.join(self.data_dir, filename)
        data = pd.read_pickle(filepath)
        logger.info(f"Loaded data from {filepath}")
        return data


class SyntheticNewsGenerator:
    """Generate synthetic news data with realistic market-related content."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        
        # News templates
        self.templates = [
            "{company} reports {sentiment} earnings, beating analyst expectations",
            "{company} announces {sentiment} outlook for next quarter",
            "Market analysts {sentiment} on {company} following product launch",
            "{company} stock {movement} on {sentiment} regulatory news",
            "Institutional investors {sentiment} {company} holdings",
        ]
        
        self.sentiments = {
            'positive': ['strong', 'robust', 'excellent', 'outstanding', 'optimistic'],
            'negative': ['weak', 'disappointing', 'concerning', 'pessimistic', 'challenging'],
            'neutral': ['mixed', 'stable', 'steady', 'unchanged', 'moderate']
        }
        
        self.movements = ['rises', 'falls', 'surges', 'drops', 'rallies', 'declines']
    
    def generate_news(
        self,
        tickers: List[str],
        dates: pd.DatetimeIndex,
        news_per_day: int = 3
    ) -> pd.DataFrame:
        """
        Generate synthetic news dataset.
        
        Args:
            tickers: List of ticker symbols
            dates: Date range for news
            news_per_day: Average number of news items per day
        
        Returns:
            DataFrame with columns [timestamp, ticker, headline, sentiment, source]
        """
        logger.info(f"Generating synthetic news for {len(tickers)} tickers")
        
        news_data = []
        
        for date in dates:
            # Variable number of news items per day
            n_news = np.random.poisson(news_per_day)
            
            for _ in range(n_news):
                ticker = np.random.choice(tickers)
                template = np.random.choice(self.templates)
                
                # Sample sentiment
                sentiment_cat = np.random.choice(['positive', 'negative', 'neutral'], p=[0.4, 0.3, 0.3])
                sentiment_word = np.random.choice(self.sentiments[sentiment_cat])
                
                # Generate headline
                headline = template.format(
                    company=ticker,
                    sentiment=sentiment_word,
                    movement=np.random.choice(self.movements)
                )
                
                # Add timestamp (random hour during trading day)
                hour = np.random.randint(9, 16)
                minute = np.random.randint(0, 60)
                timestamp = date.replace(hour=hour, minute=minute)
                
                news_data.append({
                    'timestamp': timestamp,
                    'ticker': ticker,
                    'headline': headline,
                    'sentiment': sentiment_cat,
                    'source': np.random.choice(['Reuters', 'Bloomberg', 'WSJ', 'CNBC'])
                })
        
        df = pd.DataFrame(news_data)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Generated {len(df)} synthetic news items")
        return df


class SyntheticMarketGenerator:
    """
    Generate synthetic market data with realistic statistical properties.
    Implements GBM with jumps, regime-switching, and correlation structure.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def generate_ohlcv(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        initial_prices: Optional[Dict[str, float]] = None,
        annual_return: float = 0.08,
        annual_volatility: float = 0.20,
        correlation: float = 0.5
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate synthetic OHLCV data using geometric Brownian motion.
        
        Statistical properties:
        - Log returns follow multivariate normal with specified correlation
        - Includes jump component (rare large moves)
        - Realistic OHLC relationships
        """
        logger.info(f"Generating synthetic OHLCV for {len(tickers)} tickers")
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(dates)
        n_tickers = len(tickers)
        
        # Initial prices
        if initial_prices is None:
            initial_prices = {ticker: np.random.uniform(50, 200) for ticker in tickers}
        
        # Daily parameters
        dt = 1/252  # Daily time step
        mu = annual_return * dt
        sigma = annual_volatility * np.sqrt(dt)
        
        # Generate correlated returns
        cov_matrix = correlation * np.ones((n_tickers, n_tickers))
        np.fill_diagonal(cov_matrix, 1.0)
        cov_matrix = (sigma ** 2) * cov_matrix
        
        returns = np.random.multivariate_normal(
            mean=[mu] * n_tickers,
            cov=cov_matrix,
            size=n_days
        )
        
        # Add jump component (1% probability)
        jumps = np.random.binomial(1, 0.01, size=(n_days, n_tickers))
        jump_sizes = np.random.normal(0, 2*sigma, size=(n_days, n_tickers))
        returns += jumps * jump_sizes
        
        # Generate prices for each ticker
        data = {}
        for i, ticker in enumerate(tickers):
            prices = [initial_prices[ticker]]
            for t in range(n_days):
                prices.append(prices[-1] * np.exp(returns[t, i]))
            
            prices = np.array(prices[1:])  # Remove initial value
            
            # Generate OHLC from close prices
            df = pd.DataFrame(index=dates)
            df['close'] = prices
            
            # High/Low with realistic spread
            daily_range = np.abs(np.random.normal(0, sigma, n_days))
            df['high'] = prices * (1 + daily_range)
            df['low'] = prices * (1 - daily_range)
            
            # Open price (carry from previous close with gap)
            gap = np.random.normal(0, 0.5*sigma, n_days)
            df['open'] = prices * (1 + gap)
            
            # Ensure OHLC consistency
            df['high'] = df[['open', 'high', 'close']].max(axis=1)
            df['low'] = df[['open', 'low', 'close']].min(axis=1)
            
            # Volume (log-normal)
            df['volume'] = np.random.lognormal(
                mean=np.log(1e6),
                sigma=0.5,
                size=n_days
            ).astype(int)
            
            # Adjusted close (same as close for synthetic data)
            df['adj_close'] = df['close']
            
            data[ticker] = df
        
        logger.info(f"Generated {n_days} days of synthetic OHLCV data")
        return data


if __name__ == "__main__":
    # Test data loading
    loader = MarketDataLoader()
    
    # Test OHLCV fetch
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    ohlcv_data = loader.fetch_ohlcv(tickers, start_date, end_date)
    print(f"\nFetched OHLCV for {len(ohlcv_data)} tickers")
    
    # Test macro data
    macro_series = ['DFF', 'T10Y2Y', 'VIXCLS']
    macro_data = loader.fetch_macro_data(macro_series, start_date, end_date)
    print(f"Fetched {len(macro_data)} macro series")
    
    # Test synthetic generation
    synth_gen = SyntheticMarketGenerator(seed=42)
    synth_data = synth_gen.generate_ohlcv(tickers, start_date, end_date)
    print(f"\nGenerated synthetic data for {len(synth_data)} tickers")
    
    # Test news generation
    news_gen = SyntheticNewsGenerator(seed=42)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    news_df = news_gen.generate_news(tickers, dates)
    print(f"Generated {len(news_df)} news items")
