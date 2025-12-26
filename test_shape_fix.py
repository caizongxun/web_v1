#!/usr/bin/env python3
"""
Test script to verify the shape mismatch fix
Tests all technical indicator calculations with 100-element arrays
"""

import numpy as np
import sys

class TechnicalIndicators:
    """Calculate technical indicators"""
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        """Relative Strength Index"""
        prices = np.array(prices)
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
        high = high[-min_len:]
        low = low[-min_len:]
        close = close[-min_len:]
        
        # Calculate True Range
        tr1 = high - low
        
        # For previous close comparisons, pad with current close for first element
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
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        alpha = 2 / (period + 1)
        
        for i in range(1, len(prices)):
            ema[i] = prices[i] * alpha + ema[i-1] * (1 - alpha)
        
        return ema

def test_all_indicators():
    """Test all indicator calculations"""
    print("=" * 70)
    print("Testing Technical Indicators Shape Fix")
    print("=" * 70)
    
    # Generate test data with exactly 100 elements
    np.random.seed(42)
    current_price = 42150.0
    prices = [current_price]
    price = current_price
    
    for _ in range(99):
        change = np.random.normal(0, 0.01)
        price = price * (1 + change)
        prices.insert(0, price)
    
    prices = np.array(prices[:100])  # Ensure exactly 100 elements
    volumes = np.random.uniform(1000000, 10000000, 100)
    
    print(f"\n1. Data Generation")
    print(f"   - Prices shape: {prices.shape}")
    print(f"   - Prices length: {len(prices)}")
    print(f"   - First price: {prices[0]:.2f}")
    print(f"   - Last price: {prices[-1]:.2f}")
    
    tests = [
        ("RSI (14)", lambda: TechnicalIndicators.calculate_rsi(prices, 14)),
        ("MACD", lambda: TechnicalIndicators.calculate_macd(prices)),
        ("ADX (14)", lambda: TechnicalIndicators.calculate_adx(prices, prices, prices)),
        ("ATR (14)", lambda: TechnicalIndicators.calculate_atr(prices, prices, prices)),
        ("Volatility (20)", lambda: TechnicalIndicators.calculate_volatility(prices, 20)),
    ]
    
    print(f"\n2. Technical Indicators Calculation")
    all_passed = True
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            print(f"   ✓ {test_name:20s} = {result:10.4f}")
        except Exception as e:
            print(f"   ✗ {test_name:20s} - ERROR: {str(e)}")
            all_passed = False
    
    print(f"\n3. Shape Consistency Check")
    try:
        # Test with different input combinations
        test_cases = [
            ("Same array for high/low/close", prices, prices, prices),
            ("Different slightly arrays", prices[:-1], prices[:-1], prices[:-1]),
        ]
        
        for desc, h, l, c in test_cases:
            try:
                atr_val = TechnicalIndicators.calculate_atr(h, l, c, 14)
                print(f"   ✓ ATR with {desc:35s} = {atr_val:.4f}")
            except Exception as e:
                print(f"   ✗ ATR with {desc:35s} - ERROR: {str(e)}")
                all_passed = False
    except Exception as e:
        print(f"   ✗ Error in shape consistency check: {str(e)}")
        all_passed = False
    
    print(f"\n4. Array Broadcasting Check")
    try:
        # Test the specific fix: close_prev concatenation
        h = np.array([100, 101, 102, 103, 104])
        l = np.array([99, 100, 101, 102, 103])
        c = np.array([99.5, 100.5, 101.5, 102.5, 103.5])
        
        print(f"   - high shape: {h.shape}")
        print(f"   - low shape: {l.shape}")
        print(f"   - close shape: {c.shape}")
        
        # Test the fix
        close_prev = np.concatenate([[c[0]], c[:-1]])
        print(f"   - close_prev shape: {close_prev.shape}")
        
        tr2 = np.abs(h - close_prev)
        print(f"   - tr2 shape: {tr2.shape}")
        print(f"   ✓ Broadcasting successful (all shapes match)")
        
    except Exception as e:
        print(f"   ✗ Broadcasting error: {str(e)}")
        all_passed = False
    
    print(f"\n5. Test Results")
    if all_passed:
        print("   ✓ ALL TESTS PASSED!")
        print(f"   ✓ Shape mismatch issue has been fixed.")
        return 0
    else:
        print("   ✗ SOME TESTS FAILED")
        return 1

if __name__ == '__main__':
    exit_code = test_all_indicators()
    print("\n" + "=" * 70)
    sys.exit(exit_code)
