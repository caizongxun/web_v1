#!/usr/bin/env python3
"""
Real-time Data Fetcher for Crypto Predictor
Supports: Binance API, yfinance

Author: CPB Team
Version: 1.0
"""

import requests
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import logging
from functools import lru_cache
import time

logger = logging.getLogger(__name__)

class DataFetcher:
    """Fetch real-time cryptocurrency data from multiple sources"""
    
    # Binance API configuration
    BINANCE_API_URL = "https://api.binance.com/api/v3"
    BINANCE_SPOT_URL = "https://api.binance.com/api/v3/klines"
    
    # Timeframe mapping
    TIMEFRAME_MAPPING = {
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '1h': '1h',
        '4h': '4h',
        '1d': '1d',
        '1w': '1w',
        '1M': '1M'
    }
    
    # Binance timeframe format
    BINANCE_TIMEFRAME = {
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '1h': '1h',
        '4h': '4h',
        '1d': '1d',
        '1w': '1w',
        '1M': '1M'
    }
    
    # yfinance timeframe format
    YFINANCE_TIMEFRAME = {
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '1h': '1h',
        '4h': '4h',
        '1d': '1d',
        '1w': '1wk',
        '1M': '1mo'
    }
    
    @staticmethod
    def fetch_binance_klines(symbol, timeframe, limit=100):
        """
        Fetch K-line data from Binance
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
            timeframe: Time interval (e.g., '1d', '1h', '15m')
            limit: Number of K-lines to fetch (1-1000, default 100)
        
        Returns:
            dict: Contains klines, current_price, and metadata
        """
        try:
            # Validate inputs
            if limit < 1 or limit > 1000:
                limit = min(max(limit, 1), 1000)
            
            if timeframe not in DataFetcher.BINANCE_TIMEFRAME:
                raise ValueError(f"Unsupported timeframe: {timeframe}")
            
            # Prepare parameters
            params = {
                'symbol': symbol.upper(),
                'interval': DataFetcher.BINANCE_TIMEFRAME[timeframe],
                'limit': limit
            }
            
            # Fetch data
            response = requests.get(
                DataFetcher.BINANCE_SPOT_URL,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            
            klines_raw = response.json()
            
            # Parse K-line data
            klines = []
            for kline in klines_raw:
                klines.append({
                    'timestamp': int(kline[0]),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[7]),
                    'quote_asset_volume': float(kline[8])
                })
            
            if not klines:
                raise ValueError("No data returned from Binance")
            
            # Get current price (latest close)
            current_price = float(klines[-1]['close'])
            
            # Extract price arrays
            prices = np.array([k['close'] for k in klines])
            volumes = np.array([k['volume'] for k in klines])
            highs = np.array([k['high'] for k in klines])
            lows = np.array([k['low'] for k in klines])
            opens = np.array([k['open'] for k in klines])
            
            logger.info(f"Fetched {len(klines)} candles from Binance for {symbol} {timeframe}")
            
            return {
                'source': 'binance',
                'symbol': symbol,
                'timeframe': timeframe,
                'klines': klines,
                'prices': prices,
                'volumes': volumes,
                'highs': highs,
                'lows': lows,
                'opens': opens,
                'current_price': current_price,
                'timestamp': datetime.now().isoformat(),
                'count': len(klines)
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Binance API error: {str(e)}")
            raise Exception(f"Failed to fetch data from Binance: {str(e)}")
        except (ValueError, KeyError) as e:
            logger.error(f"Data parsing error: {str(e)}")
            raise Exception(f"Failed to parse Binance data: {str(e)}")
    
    @staticmethod
    def fetch_yfinance_data(symbol, period='100d', interval='1d'):
        """
        Fetch K-line data from yfinance (Yahoo Finance)
        
        Args:
            symbol: Trading symbol (e.g., 'BTC-USD', 'ETH-USD')
            period: Data period (e.g., '100d', '1y')
            interval: Timeframe (e.g., '1d', '1h', '15m')
        
        Returns:
            dict: Contains klines, current_price, and metadata
        """
        try:
            # Download data from yfinance
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                raise ValueError(f"No data returned for {symbol}")
            
            # Convert to list of dicts
            klines = []
            for idx, row in df.iterrows():
                klines.append({
                    'timestamp': int(idx.timestamp() * 1000),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': float(row['Volume'])
                })
            
            # Get current price
            current_price = float(klines[-1]['close'])
            
            # Extract arrays
            prices = np.array([k['close'] for k in klines])
            volumes = np.array([k['volume'] for k in klines])
            highs = np.array([k['high'] for k in klines])
            lows = np.array([k['low'] for k in klines])
            opens = np.array([k['open'] for k in klines])
            
            logger.info(f"Fetched {len(klines)} candles from yfinance for {symbol}")
            
            return {
                'source': 'yfinance',
                'symbol': symbol,
                'period': period,
                'interval': interval,
                'klines': klines,
                'prices': prices,
                'volumes': volumes,
                'highs': highs,
                'lows': lows,
                'opens': opens,
                'current_price': current_price,
                'timestamp': datetime.now().isoformat(),
                'count': len(klines)
            }
            
        except Exception as e:
            logger.error(f"yfinance error: {str(e)}")
            raise Exception(f"Failed to fetch data from yfinance: {str(e)}")
    
    @staticmethod
    def get_crypto_data(symbol, timeframe='1d', limit=100, source='binance'):
        """
        Unified interface to fetch crypto data from any source
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: Time interval (e.g., '1d', '1h', '15m')
            limit: Number of K-lines to fetch
            source: Data source ('binance' or 'yfinance')
        
        Returns:
            dict: Unified data format
        """
        if source.lower() == 'binance':
            # For Binance, symbol should be like 'BTCUSDT'
            return DataFetcher.fetch_binance_klines(symbol, timeframe, limit)
        
        elif source.lower() == 'yfinance':
            # For yfinance, symbol should be like 'BTC-USD'
            # Auto-convert if needed
            if 'USDT' in symbol:
                symbol = symbol.replace('USDT', '-USD')
            elif symbol.endswith('USD'):
                symbol = symbol[:-3] + '-USD'
            
            # Map timeframe to period
            period_map = {
                '1d': 'max',
                '1h': '60d',
                '15m': '7d',
                '5m': '7d',
                '1m': '1d'
            }
            period = period_map.get(timeframe, '100d')
            
            return DataFetcher.fetch_yfinance_data(symbol, period, timeframe)
        
        else:
            raise ValueError(f"Unsupported source: {source}")
    
    @staticmethod
    def validate_data(data_dict, min_klines=30):
        """
        Validate fetched data quality
        
        Args:
            data_dict: Data dictionary from fetch functions
            min_klines: Minimum number of K-lines required
        
        Returns:
            bool: True if data is valid
        """
        try:
            # Check required fields
            required_fields = ['prices', 'volumes', 'highs', 'lows', 'current_price']
            for field in required_fields:
                if field not in data_dict:
                    logger.warning(f"Missing field: {field}")
                    return False
            
            # Check data length
            if len(data_dict['prices']) < min_klines:
                logger.warning(f"Insufficient data: {len(data_dict['prices'])} < {min_klines}")
                return False
            
            # Check for NaN values
            if np.any(np.isnan(data_dict['prices'])):
                logger.warning("NaN values found in prices")
                return False
            
            # Check prices are positive
            if np.any(data_dict['prices'] <= 0):
                logger.warning("Non-positive prices detected")
                return False
            
            # Check high >= close >= low
            for i in range(len(data_dict['klines'])):
                h = data_dict['highs'][i]
                l = data_dict['lows'][i]
                c = data_dict['prices'][i]
                if not (h >= c >= l):
                    logger.warning(f"Invalid OHLC at index {i}: H={h}, C={c}, L={l}")
                    return False
            
            logger.info("Data validation passed")
            return True
        
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False


class DataCache:
    """Simple cache for API responses to avoid rate limiting"""
    
    def __init__(self, ttl_seconds=300):
        self.cache = {}
        self.ttl = ttl_seconds
        self.timestamps = {}
    
    def get(self, key):
        """Get cached data if not expired"""
        if key in self.cache:
            if time.time() - self.timestamps[key] < self.ttl:
                logger.info(f"Cache hit: {key}")
                return self.cache[key]
            else:
                # Expired
                del self.cache[key]
                del self.timestamps[key]
        return None
    
    def set(self, key, value):
        """Store data in cache"""
        self.cache[key] = value
        self.timestamps[key] = time.time()
        logger.info(f"Cache set: {key}")
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.timestamps.clear()
        logger.info("Cache cleared")


# Global cache instance
data_cache = DataCache(ttl_seconds=300)  # 5 minutes cache


if __name__ == '__main__':
    # Test the data fetcher
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Binance data fetching...")
    try:
        data = DataFetcher.get_crypto_data('BTCUSDT', '1d', 100, 'binance')
        print(f"✓ Binance: {data['count']} candles, Current price: {data['current_price']}")
        print(f"  Prices: {data['prices'][-5:]}")
        print(f"  Valid: {DataFetcher.validate_data(data)}")
    except Exception as e:
        print(f"✗ Binance error: {e}")
    
    print("\nTesting yfinance data fetching...")
    try:
        data = DataFetcher.get_crypto_data('BTCUSDT', '1d', 100, 'yfinance')
        print(f"✓ yfinance: {data['count']} candles, Current price: {data['current_price']}")
        print(f"  Prices: {data['prices'][-5:]}")
        print(f"  Valid: {DataFetcher.validate_data(data)}")
    except Exception as e:
        print(f"✗ yfinance error: {e}")
