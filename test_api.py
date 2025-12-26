#!/usr/bin/env python3
"""
CPB Crypto Predictor V6 - API Test Suite
模型預測 API 測試
"""

import requests
import json
import time
from datetime import datetime

# Configuration
API_BASE_URL = 'http://localhost:8001'
API_ENDPOINTS = {
    'predict': '/api/v6/predict',
    'symbols': '/api/v6/symbols',
    'config': '/api/v6/config',
    'health': '/api/v6/health'
}

class bcolors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    """Print formatted header"""
    print(f"\n{bcolors.BOLD}{bcolors.HEADER}{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}{bcolors.ENDC}\n")

def print_success(text):
    """Print success message"""
    print(f"{bcolors.OKGREEN}✓ {text}{bcolors.ENDC}")

def print_error(text):
    """Print error message"""
    print(f"{bcolors.FAIL}× {text}{bcolors.ENDC}")

def print_info(text):
    """Print info message"""
    print(f"{bcolors.OKCYAN}i {text}{bcolors.ENDC}")

def test_health():
    """Test health check endpoint"""
    print_header("Testing Health Check")
    
    try:
        response = requests.get(f"{API_BASE_URL}{API_ENDPOINTS['health']}", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print_success("Health check passed")
            print_info(f"Status: {data['status']}")
            print_info(f"Version: {data['version']}")
            print_info(f"Timestamp: {data['timestamp']}")
            return True
        else:
            print_error(f"Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Health check error: {str(e)}")
        return False

def test_symbols():
    """Test symbols endpoint"""
    print_header("Testing Symbols Endpoint")
    
    try:
        response = requests.get(f"{API_BASE_URL}{API_ENDPOINTS['symbols']}", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print_success("Symbols endpoint responded")
            print_info(f"Full Support: {len(data['full_support'])} cryptocurrencies")
            print_info(f"Partial Support: {len(data['partial_support'])} cryptocurrencies")
            print_info(f"Timeframes: {', '.join(data['timeframes'])}")
            
            # Display accuracy levels
            print(f"\n{bcolors.OKCYAN}Accuracy Levels:{bcolors.ENDC}")
            for tf, accuracy in data['accuracy'].items():
                print(f"  {tf}: {accuracy*100:.1f}%")
            
            return True
        else:
            print_error(f"Symbols endpoint failed with status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Symbols endpoint error: {str(e)}")
        return False

def test_config():
    """Test model config endpoint"""
    print_header("Testing Model Configuration")
    
    try:
        response = requests.get(f"{API_BASE_URL}{API_ENDPOINTS['config']}", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print_success("Model configuration retrieved")
            print_info(f"Version: {data['version']}")
            print_info(f"LSTM Weight: {data['lstm_weight']*100:.1f}%")
            print_info(f"GRU Weight: {data['gru_weight']*100:.1f}%")
            print_info(f"XGBoost Weight: {data['xgboost_weight']*100:.1f}%")
            
            print(f"\n{bcolors.OKCYAN}LSTM Configuration:{bcolors.ENDC}")
            print(f"  Layers: {data['lstm_layers']}")
            
            print(f"\n{bcolors.OKCYAN}GRU Configuration:{bcolors.ENDC}")
            print(f"  Layers: {data['gru_layers']}")
            
            print(f"\n{bcolors.OKCYAN}XGBoost Configuration:{bcolors.ENDC}")
            print(f"  Tree Depth: {data['xgboost_depth']}")
            print(f"  Learning Rate: {data['xgboost_learning_rate']}")
            print(f"  Estimators: {data['xgboost_n_estimators']}")
            
            return True
        else:
            print_error(f"Config endpoint failed with status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Config endpoint error: {str(e)}")
        return False

def test_prediction(symbol, timeframe, klines):
    """Test prediction endpoint"""
    print_header(f"Testing Prediction: {symbol} {timeframe}")
    
    payload = {
        'symbol': symbol,
        'timeframe': timeframe,
        'klines': klines
    }
    
    print_info(f"Request: {json.dumps(payload)}")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}{API_ENDPOINTS['predict']}",
            json=payload,
            timeout=30
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Prediction completed in {elapsed:.2f}s")
            
            # Display current price
            print(f"\n{bcolors.OKCYAN}Current Market Data:{bcolors.ENDC}")
            print(f"  Current Price: ${data['current_price']:.8f}")
            print(f"  K-lines Analyzed: {data['klines_count']}")
            
            # Display prediction
            print(f"\n{bcolors.OKCYAN}Prediction:{bcolors.ENDC}")
            print(f"  Predicted Price: ${data['predicted_price']:.8f}")
            change = ((data['predicted_price'] - data['current_price']) / data['current_price'] * 100)
            color = bcolors.OKGREEN if change > 0 else bcolors.FAIL
            print(f"  Price Change: {color}{change:+.2f}%{bcolors.ENDC}")
            print(f"  Accuracy: {data['accuracy']*100:.1f}%")
            
            # Display confidence and signal
            print(f"\n{bcolors.OKCYAN}Trading Signal:{bcolors.ENDC}")
            print(f"  Confidence: {data['confidence']*100:.1f}%")
            print(f"  Recommendation: {data['recommendation']}")
            
            # Display risk management
            print(f"\n{bcolors.OKCYAN}Risk Management:{bcolors.ENDC}")
            print(f"  Entry Price: ${data['entry_price']:.8f}")
            print(f"  Stop Loss: ${data['stop_loss']:.8f}")
            print(f"  Take Profit: ${data['take_profit']:.8f}")
            
            # Display technical indicators
            print(f"\n{bcolors.OKCYAN}Technical Indicators:{bcolors.ENDC}")
            for indicator, value in data['technical_indicators'].items():
                print(f"  {indicator}: {value:.2f}")
            
            # Display volatility
            print(f"\n{bcolors.OKCYAN}Volatility Assessment:{bcolors.ENDC}")
            print(f"  Current: {data['volatility']['current']*100:.3f}%")
            print(f"  Predicted: {data['volatility']['predicted']*100:.3f}%")
            print(f"  Level: {data['volatility']['level']}")
            
            # Display model predictions
            print(f"\n{bcolors.OKCYAN}Model Predictions:{bcolors.ENDC}")
            for model, pred in data['model_predictions'].items():
                print(f"  {model}: ${pred:.8f}")
            
            return True
        else:
            print_error(f"Prediction failed with status {response.status_code}")
            print_error(f"Response: {response.text}")
            return False
    except Exception as e:
        print_error(f"Prediction error: {str(e)}")
        return False

def run_comprehensive_tests():
    """Run comprehensive test suite"""
    print(f"{bcolors.BOLD}{bcolors.HEADER}")
    print("  CPB Crypto Predictor V6 - API Test Suite")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{bcolors.ENDC}")
    
    results = []
    
    # Test 1: Health Check
    results.append(("Health Check", test_health()))
    
    # Test 2: Symbols
    results.append(("Symbols Endpoint", test_symbols()))
    
    # Test 3: Model Config
    results.append(("Model Configuration", test_config()))
    
    # Test 4: Predictions with different cryptocurrencies and timeframes
    test_cases = [
        ('BTCUSDT', '1d', 100),  # Bitcoin, daily
        ('ETHUSDT', '1h', 48),   # Ethereum, hourly
        ('BNBUSDT', '15m', 96),  # BNB, 15-minute
    ]
    
    for symbol, timeframe, klines in test_cases:
        results.append(
            (f"Prediction {symbol} {timeframe}", test_prediction(symbol, timeframe, klines))
        )
        time.sleep(1)  # Rate limiting
    
    # Print summary
    print_header("Test Summary")
    total = len(results)
    passed = sum(1 for _, result in results if result)
    
    for test_name, result in results:
        if result:
            print_success(test_name)
        else:
            print_error(test_name)
    
    print(f"\n{bcolors.BOLD}Total: {passed}/{total} tests passed{bcolors.ENDC}")
    
    if passed == total:
        print(f"{bcolors.OKGREEN}\nAll tests passed!✓{bcolors.ENDC}")
        return 0
    else:
        print(f"{bcolors.FAIL}\nSome tests failed!{bcolors.ENDC}")
        return 1

if __name__ == '__main__':
    import sys
    sys.exit(run_comprehensive_tests())
