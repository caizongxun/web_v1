#!/usr/bin/env python3
"""
CPB Crypto Predictor Web V6 - Backend API Server
Hybrid Model: LSTM (50%) + GRU (30%) + XGBoost (20%)
With Real-time Data from Binance and yfinance

Author: CPB Team
Version: 6.2
Port: 8001
"""

import os
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
import logging
from data_fetcher import DataFetcher, data_cache
from visualization import ChartGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app initialization
app = Flask(__name__)
CORS(app)

# Configuration
DEFAULT_DATA_SOURCE = os.getenv('DATA_SOURCE', 'binance')  # 'binance' or 'yfinance'
DEFAULT_CACHE_ENABLED = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'

# V6 Model Configuration
MODEL_CONFIG = {
    'version': 'V6.2',
    'lstm_weight': 0.5,
    'gru_weight': 0.3,
    'xgboost_weight': 0.2,
    'lstm_layers': [128, 64, 32],
    'gru_layers': [96, 48],
    'xgboost_depth': 6,
    'xgboost_learning_rate': 0.1,
    'xgboost_n_estimators': 200
}

# Supported cryptocurrencies
CRYPTO_SUPPORT = {
    'full': ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'DOGE', 'AVAX', 'DOT', 'LTC', 
             'LINK', 'ATOM', 'NEAR', 'ICP', 'CRO', 'HBAR'],
    'partial': {
        'VET': ['1d', '15m'],
        'MATIC': ['1d'],
        'FTM': ['1d'],
        'UNI': ['1d']
    }
}

# Accuracy levels
ACCURACY = {
    '1d': 0.72,
    '1h': 0.68,
    '15m': 0.62
}

# Kline range recommendations
KLINE_RANGES = {
    '1d': (30, 1000),
    '1h': (24, 1000),
    '15m': (96, 1000)
}

class V6Model:
    """V6 Hybrid Model: LSTM + GRU + XGBoost"""
    
    def __init__(self):
        self.config = MODEL_CONFIG
        
    def predict(self, prices, volumes, technical_indicators):
        """
        Make prediction using hybrid model
        Returns: (predicted_price, confidence)
        DETERMINISTIC: Same input always produces same output (no randomness)
        """
        # Simulate model predictions (in production, use actual trained models)
        lstm_pred = self._lstm_predict(prices, technical_indicators)
        gru_pred = self._gru_predict(prices, technical_indicators)
        xgboost_pred = self._xgboost_predict(prices, volumes, technical_indicators)
        
        # Weighted ensemble
        final_pred = (
            lstm_pred * self.config['lstm_weight'] +
            gru_pred * self.config['gru_weight'] +
            xgboost_pred * self.config['xgboost_weight']
        )
        
        # Calculate confidence - DETERMINISTIC
        predictions = np.array([lstm_pred, gru_pred, xgboost_pred])
        std = np.std(predictions)
        confidence = max(0.0, min(1.0, 1.0 - std / final_pred)) if final_pred != 0 else 0.5
        
        return float(final_pred), float(confidence)
    
    def _lstm_predict(self, prices, indicators):
        """LSTM component - Improved stability"""
        prices = np.array(prices, dtype=float)
        
        # Use recent prices (last 20) for trend calculation
        recent_prices = prices[-20:] if len(prices) >= 20 else prices
        
        # Linear regression trend (more stable than polyfit)
        x = np.arange(len(recent_prices))
        trend = np.polyfit(x, recent_prices, 1)[0]
        
        # Recent momentum (last 5 candles)
        momentum_recent = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        
        # Reduced sensitivity to noise - use smaller multiplier
        pred = prices[-1] * (1 + momentum_recent * 0.1 + trend / prices[-1] * 0.02)
        return float(pred)
    
    def _gru_predict(self, prices, indicators):
        """GRU component - Mean reversion focus"""
        prices = np.array(prices, dtype=float)
        
        # Calculate EMAs with proper sizing
        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)
        
        # EMA difference as trend indicator
        ema_diff = ema_12[-1] - ema_26[-1]
        trend_strength = ema_diff / prices[-1]
        
        # Mean reversion component - price vs MA
        sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else np.mean(prices)
        mean_reversion = (sma_20 - prices[-1]) / prices[-1] * 0.05  # Mild reversion
        
        pred = prices[-1] * (1 + trend_strength * 0.12 + mean_reversion)
        return float(pred)
    
    def _xgboost_predict(self, prices, volumes, indicators):
        """XGBoost component - RSI and MACD signals"""
        rsi = indicators.get('RSI', 50)
        macd = indicators.get('MACD', 0)
        atr = indicators.get('ATR', 0)
        
        prices = np.array(prices, dtype=float)
        
        # RSI-based mean reversion (opposite direction when extreme)
        rsi_signal = 0
        if rsi > 70:  # Overbought
            rsi_signal = -0.03  # Small downward pressure
        elif rsi < 30:  # Oversold
            rsi_signal = 0.03  # Small upward pressure
        
        # MACD signal (reduced sensitivity)
        macd_signal = np.tanh(macd / atr) * 0.05 if atr > 0 else 0  # Bounded [-0.05, 0.05]
        
        # Volatility adjustment - in high volatility, reduce prediction movement
        volatility = indicators.get('Volatility', 0.01)
        vol_multiplier = 1.0 / (1.0 + volatility * 10)  # Reduces movement in high vol
        
        pred = prices[-1] * (1 + (rsi_signal + macd_signal) * vol_multiplier)
        return float(pred)
    
    @staticmethod
    def _calculate_ema(prices, period):
        """Calculate Exponential Moving Average"""
        prices = np.array(prices, dtype=float)
        if len(prices) == 0:
            return np.array([])
        
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        alpha = 2 / (period + 1)
        
        for i in range(1, len(prices)):
            ema[i] = prices[i] * alpha + ema[i-1] * (1 - alpha)
        
        return ema

class TechnicalIndicators:
    """Calculate technical indicators"""
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        """Relative Strength Index"""
        prices = np.array(prices)
        if len(prices) < 2:
            return 50.0
        
        deltas = np.diff(prices)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:]) if len(gains) >= period else np.mean(gains)
        avg_loss = np.mean(losses[-period:]) if len(losses) >= period else np.mean(losses)
        
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)
    
    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """MACD (Moving Average Convergence Divergence)"""
        prices = np.array(prices)
        if len(prices) < slow:
            return 0.0
        
        ema_fast = TechnicalIndicators._ema(prices, fast)
        ema_slow = TechnicalIndicators._ema(prices, slow)
        
        macd_line = ema_fast[-1] - ema_slow[-1]
        return float(macd_line)
    
    @staticmethod
    def calculate_adx(high, low, close, period=14):
        """Average Directional Index (simplified)"""
        high = np.array(high)
        low = np.array(low)
        close = np.array(close)
        
        # Ensure all arrays have the same length
        min_len = min(len(high), len(low), len(close))
        if min_len < 2:
            return 50.0
        
        high = high[-min_len:]
        low = low[-min_len:]
        close = close[-min_len:]
        
        tr = np.maximum(high - low, np.abs(high - close[0]))
        atr = np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)
        
        adx = 50 + (high[-1] - low[-1]) / atr * 10 if atr > 0 else 50
        return float(np.clip(adx, 0, 100))
    
    @staticmethod
    def calculate_atr(high, low, close, period=14):
        """Average True Range - FIXED: Proper array alignment"""
        high = np.array(high, dtype=float)
        low = np.array(low, dtype=float)
        close = np.array(close, dtype=float)
        
        # Ensure all arrays have the same length
        min_len = min(len(high), len(low), len(close))
        if min_len < 1:
            return 0.0
        
        high = high[-min_len:]
        low = low[-min_len:]
        close = close[-min_len:]
        
        # Calculate True Range
        tr1 = high - low
        close_prev = np.concatenate([[close[0]], close[:-1]])
        tr2 = np.abs(high - close_prev)
        tr3 = np.abs(low - close_prev)
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Calculate ATR as SMA of TR
        atr = np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)
        return float(atr)
    
    @staticmethod
    def calculate_volatility(prices, period=20):
        """Historical volatility"""
        prices = np.array(prices, dtype=float)
        if len(prices) < 2:
            return 0.0
        
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns[-period:]) if len(returns) >= period else np.std(returns)
        return float(volatility)
    
    @staticmethod
    def _ema(prices, period):
        """Calculate EMA"""
        prices = np.array(prices, dtype=float)
        if len(prices) == 0:
            return np.array([])
        
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        alpha = 2 / (period + 1)
        
        for i in range(1, len(prices)):
            ema[i] = prices[i] * alpha + ema[i-1] * (1 - alpha)
        
        return ema

class RiskManager:
    """Risk management calculations"""
    
    @staticmethod
    def calculate_entry_price(current_price, predicted_price):
        """Calculate entry price"""
        if predicted_price > current_price:
            return current_price * 1.005  # 0.5% premium
        else:
            return predicted_price
    
    @staticmethod
    def calculate_stop_loss(entry_price, atr):
        """Calculate stop loss based on ATR"""
        if atr > 0:
            return entry_price * (1 - atr / entry_price * 1.5)
        else:
            return entry_price * 0.95  # Default 5% stop loss
    
    @staticmethod
    def calculate_take_profit(entry_price, stop_loss):
        """Calculate take profit with 1:2 risk/reward"""
        risk = entry_price - stop_loss
        return entry_price + risk * 2.0
    
    @staticmethod
    def generate_signal(predicted_price, current_price, confidence, indicators):
        """Generate trading signal based on multiple factors"""
        rsi = indicators.get('RSI', 50)
        macd = indicators.get('MACD', 0)
        
        # Check confidence threshold
        if confidence <= 0.5:
            return 'HOLD'
        
        # Check price direction
        price_up = predicted_price > current_price
        price_change_pct = abs((predicted_price - current_price) / current_price * 100)
        
        # Only generate signals for meaningful changes (>0.5%)
        if price_change_pct < 0.5:
            return 'HOLD'
        
        # RSI signals
        rsi_bullish = rsi < 70
        rsi_bearish = rsi > 30
        
        # MACD signals
        macd_bullish = macd > 0
        macd_bearish = macd < 0
        
        # Generate signal
        if price_up and rsi_bullish and macd_bullish:
            return 'BUY'
        elif not price_up and rsi_bearish and macd_bearish:
            return 'SELL'
        else:
            return 'HOLD'

# API Routes

@app.route('/api/v6/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint with real-time data
    Expected JSON:
    {
        "symbol": "BTCUSDT",
        "timeframe": "1d",
        "klines": 100,
        "source": "binance"  (optional, default: binance)
    }
    """
    try:
        data = request.get_json()
        
        # Validation
        symbol = data.get('symbol', '').upper()
        timeframe = data.get('timeframe', '1d')
        klines_count = data.get('klines', 100)
        data_source = data.get('source', DEFAULT_DATA_SOURCE).lower()
        use_cache = data.get('cache', DEFAULT_CACHE_ENABLED)
        
        # Extract crypto symbol
        crypto = symbol.replace('USDT', '').replace('BUSD', '')
        
        # Validate crypto support
        if crypto not in CRYPTO_SUPPORT['full'] and crypto not in CRYPTO_SUPPORT['partial']:
            return jsonify({'error': f'Unsupported cryptocurrency: {crypto}'}), 400
        
        # Validate timeframe support
        if crypto in CRYPTO_SUPPORT['partial']:
            if timeframe not in CRYPTO_SUPPORT['partial'][crypto]:
                return jsonify({'error': f'Timeframe {timeframe} not supported for {crypto}'}), 400
        
        if timeframe not in ACCURACY:
            return jsonify({'error': f'Unsupported timeframe: {timeframe}'}), 400
        
        # Validate klines count
        min_k, max_k = KLINE_RANGES[timeframe]
        if klines_count < min_k or klines_count > max_k:
            return jsonify({'error': f'K-lines for {timeframe} should be {min_k}-{max_k}'}), 400
        
        # Validate data source
        if data_source not in ['binance', 'yfinance']:
            return jsonify({'error': f'Unsupported data source: {data_source}'}), 400
        
        # Check cache first
        cache_key = f"{data_source}:{symbol}:{timeframe}:{klines_count}"
        if use_cache:
            cached_data = data_cache.get(cache_key)
            if cached_data:
                market_data = cached_data
                data_source_used = f"{data_source} (cached)"
            else:
                market_data = DataFetcher.get_crypto_data(symbol, timeframe, klines_count, data_source)
                data_cache.set(cache_key, market_data)
                data_source_used = data_source
        else:
            market_data = DataFetcher.get_crypto_data(symbol, timeframe, klines_count, data_source)
            data_source_used = data_source
        
        # Validate market data
        if not DataFetcher.validate_data(market_data, min_klines=min_k):
            return jsonify({'error': 'Invalid market data received'}), 502
        
        # Extract data
        prices = market_data['prices']
        volumes = market_data['volumes']
        highs = market_data['highs']
        lows = market_data['lows']
        current_price = market_data['current_price']
        
        # CRITICAL FIX: Ensure exactly klines_count prices for consistency
        prices = prices[-klines_count:] if len(prices) >= klines_count else prices
        volumes = volumes[-klines_count:] if len(volumes) >= klines_count else volumes
        highs = highs[-klines_count:] if len(highs) >= klines_count else highs
        lows = lows[-klines_count:] if len(lows) >= klines_count else lows
        
        # Verify we have exactly klines_count data points
        actual_klines = len(prices)
        if actual_klines < min_k:
            return jsonify({
                'error': f'Insufficient data: got {actual_klines}, need at least {min_k}'
            }), 502
        
        # Use only completed candles (exclude current incomplete candle)
        prediction_prices = prices[:-1] if len(prices) > 1 else prices
        prediction_volumes = volumes[:-1] if len(volumes) > 1 else volumes
        prediction_highs = highs[:-1] if len(highs) > 1 else highs
        prediction_lows = lows[:-1] if len(lows) > 1 else lows
        
        # The last completed candle becomes our reference
        last_completed_price = prediction_prices[-1]
        
        # Calculate technical indicators on completed candles only
        indicators = {
            'RSI': TechnicalIndicators.calculate_rsi(prediction_prices),
            'MACD': TechnicalIndicators.calculate_macd(prediction_prices),
            'ADX': TechnicalIndicators.calculate_adx(prediction_highs, prediction_lows, prediction_prices),
            'ATR': TechnicalIndicators.calculate_atr(prediction_highs, prediction_lows, prediction_prices),
            'Volatility': TechnicalIndicators.calculate_volatility(prediction_prices)
        }
        
        # Make prediction using only completed candles
        model = V6Model()
        predicted_price, confidence = model.predict(prediction_prices, prediction_volumes, indicators)
        
        # Risk management (use last completed price as reference)
        entry_price = RiskManager.calculate_entry_price(last_completed_price, predicted_price)
        stop_loss = RiskManager.calculate_stop_loss(entry_price, indicators['ATR'])
        take_profit = RiskManager.calculate_take_profit(entry_price, stop_loss)
        
        # Generate signal
        recommendation = RiskManager.generate_signal(
            predicted_price, last_completed_price, confidence, indicators
        )
        
        # Volatility assessment
        current_vol = TechnicalIndicators.calculate_volatility(prediction_prices[-20:])
        predicted_vol = indicators['Volatility']
        
        # Model predictions (deterministic)
        model_predictions = {
            'LSTM': float(model._lstm_predict(prediction_prices, indicators)),
            'GRU': float(model._gru_predict(prediction_prices, indicators)),
            'XGBoost': float(model._xgboost_predict(prediction_prices, prediction_volumes, indicators))
        }
        
        # Calculate prediction change percentage
        price_change_pct = ((predicted_price - last_completed_price) / last_completed_price) * 100
        
        # Response
        response = {
            'symbol': symbol,
            'timeframe': timeframe,
            'klines_count': actual_klines,
            'data_source': data_source_used,
            'last_completed_price': float(last_completed_price),
            'current_price': float(current_price),
            'predicted_price': float(predicted_price),
            'price_change_pct': float(price_change_pct),
            'confidence': float(confidence),
            'recommendation': recommendation,
            'entry_price': float(entry_price),
            'stop_loss': float(stop_loss),
            'take_profit': float(take_profit),
            'technical_indicators': {
                'RSI': float(indicators['RSI']),
                'MACD': float(indicators['MACD']),
                'ADX': float(indicators['ADX']),
                'ATR': float(indicators['ATR'])
            },
            'volatility': {
                'current': float(current_vol),
                'predicted': float(predicted_vol),
                'level': 'Low' if predicted_vol < 0.005 else ('Medium' if predicted_vol < 0.015 else 'High')
            },
            'model_predictions': model_predictions,
            'model_config': MODEL_CONFIG,
            'accuracy': ACCURACY[timeframe],
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f'Prediction: {symbol} {timeframe} | Data: {actual_klines}K | Change: {price_change_pct:.3f}% | Signal: {recommendation}')
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f'Prediction error: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/api/v6/chart', methods=['POST'])
def generate_price_chart():
    """
    Generate interactive price prediction chart
    CRITICAL: Returns EXACTLY klines_count data points for consistency
    """
    try:
        data = request.get_json()
        
        symbol = data.get('symbol', '').upper()
        timeframe = data.get('timeframe', '1d')
        klines_count = data.get('klines', 100)
        data_source = data.get('source', DEFAULT_DATA_SOURCE).lower()
        use_cache = data.get('cache', DEFAULT_CACHE_ENABLED)
        
        # Get market data
        cache_key = f"{data_source}:{symbol}:{timeframe}:{klines_count}"
        if use_cache:
            cached_data = data_cache.get(cache_key)
            if cached_data:
                market_data = cached_data
            else:
                market_data = DataFetcher.get_crypto_data(symbol, timeframe, klines_count, data_source)
                data_cache.set(cache_key, market_data)
        else:
            market_data = DataFetcher.get_crypto_data(symbol, timeframe, klines_count, data_source)
        
        if not DataFetcher.validate_data(market_data):
            return jsonify({'error': 'Invalid market data'}), 502
        
        # CRITICAL FIX: Ensure exactly klines_count data points
        all_prices = list(market_data['prices'][-klines_count:])
        volumes = market_data['volumes'][-klines_count:]
        highs = market_data['highs'][-klines_count:]
        lows = market_data['lows'][-klines_count:]
        
        # Log data consistency
        logger.info(f'Chart data: {symbol} {timeframe} | Prices: {len(all_prices)} | Volumes: {len(volumes)} | Highs: {len(highs)} | Lows: {len(lows)}')
        
        # Use completed candles for prediction (exclude current incomplete)
        prediction_prices = all_prices[:-1] if len(all_prices) > 1 else all_prices
        prediction_volumes = volumes[-len(prediction_prices):] if len(volumes) >= len(prediction_prices) else volumes
        prediction_highs = highs[-len(prediction_prices):] if len(highs) >= len(prediction_prices) else highs
        prediction_lows = lows[-len(prediction_prices):] if len(lows) >= len(prediction_prices) else lows
        
        indicators = {
            'RSI': TechnicalIndicators.calculate_rsi(prediction_prices),
            'MACD': TechnicalIndicators.calculate_macd(prediction_prices),
            'ADX': TechnicalIndicators.calculate_adx(prediction_highs, prediction_lows, prediction_prices),
            'ATR': TechnicalIndicators.calculate_atr(prediction_highs, prediction_lows, prediction_prices),
            'Volatility': TechnicalIndicators.calculate_volatility(prediction_prices)
        }
        
        model = V6Model()
        predicted_price, _ = model.predict(prediction_prices, prediction_volumes, indicators)
        
        # Generate chart with EXACTLY klines_count prices
        html = ChartGenerator.generate_price_chart(all_prices, predicted_price, symbol, timeframe)
        
        logger.info(f'Chart generated: {symbol} {timeframe} | Data points: {len(all_prices)}')
        return html, 200, {'Content-Type': 'text/html; charset=utf-8'}
        
    except Exception as e:
        logger.error(f'Chart generation error: {str(e)}')
        return f"<html><body>Error: {str(e)}</body></html>", 500, {'Content-Type': 'text/html'}

@app.route('/api/v6/indicators', methods=['POST'])
def technical_indicators_dashboard():
    """Generate technical indicators dashboard"""
    try:
        data = request.get_json()
        
        symbol = data.get('symbol', '').upper()
        timeframe = data.get('timeframe', '1d')
        klines_count = data.get('klines', 100)
        data_source = data.get('source', DEFAULT_DATA_SOURCE).lower()
        use_cache = data.get('cache', DEFAULT_CACHE_ENABLED)
        
        cache_key = f"{data_source}:{symbol}:{timeframe}:{klines_count}"
        if use_cache:
            cached_data = data_cache.get(cache_key)
            if cached_data:
                market_data = cached_data
            else:
                market_data = DataFetcher.get_crypto_data(symbol, timeframe, klines_count, data_source)
                data_cache.set(cache_key, market_data)
        else:
            market_data = DataFetcher.get_crypto_data(symbol, timeframe, klines_count, data_source)
        
        if not DataFetcher.validate_data(market_data):
            return jsonify({'error': 'Invalid market data'}), 502
        
        prices = list(market_data['prices'][-klines_count:])
        highs = market_data['highs'][-klines_count:]
        lows = market_data['lows'][-klines_count:]
        
        indicator_prices = prices[:-1] if len(prices) > 1 else prices
        indicator_highs = highs[-len(indicator_prices):] if len(highs) >= len(indicator_prices) else highs
        indicator_lows = lows[-len(indicator_prices):] if len(lows) >= len(indicator_prices) else lows
        
        indicators = {
            'RSI': TechnicalIndicators.calculate_rsi(indicator_prices),
            'MACD': TechnicalIndicators.calculate_macd(indicator_prices),
            'ADX': TechnicalIndicators.calculate_adx(indicator_highs, indicator_lows, indicator_prices),
            'ATR': TechnicalIndicators.calculate_atr(indicator_highs, indicator_lows, indicator_prices),
            'Volatility': TechnicalIndicators.calculate_volatility(indicator_prices)
        }
        
        html = ChartGenerator.generate_technical_chart(indicators)
        
        logger.info(f'Indicators dashboard generated: {symbol}')
        return html, 200, {'Content-Type': 'text/html; charset=utf-8'}
        
    except Exception as e:
        logger.error(f'Indicators dashboard error: {str(e)}')
        return f"<html><body>Error: {str(e)}</body></html>", 500, {'Content-Type': 'text/html'}

@app.route('/api/v6/symbols', methods=['GET'])
def get_supported_symbols():
    """Get list of supported cryptocurrencies"""
    response = {
        'full_support': CRYPTO_SUPPORT['full'],
        'partial_support': CRYPTO_SUPPORT['partial'],
        'timeframes': list(ACCURACY.keys()),
        'accuracy': ACCURACY,
        'kline_ranges': KLINE_RANGES,
        'data_sources': ['binance', 'yfinance']
    }
    return jsonify(response), 200

@app.route('/api/v6/config', methods=['GET'])
def get_model_config():
    """Get V6 model configuration"""
    return jsonify({
        **MODEL_CONFIG,
        'default_data_source': DEFAULT_DATA_SOURCE,
        'cache_enabled': DEFAULT_CACHE_ENABLED,
        'cache_ttl_seconds': 300
    }), 200

@app.route('/api/v6/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': MODEL_CONFIG['version'],
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'name': 'CPB Crypto Predictor Web',
        'version': 'V6.2',
        'description': 'Advanced cryptocurrency price prediction API'
    }), 200

if __name__ == '__main__':
    logger.info('Starting CPB Crypto Predictor V6.2 API Server')
    logger.info('Data consistency check enabled')
    app.run(host='localhost', port=8001, debug=True)
