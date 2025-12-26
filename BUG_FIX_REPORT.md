# Shape Mismatch Error Fix Report

## 問題描述

**錯誤訊息：**
```
operands could not be broadcast together with shapes (100,) (99,)
```

**發生位置：** `/api/v6/predict` 端點

**根本原因：** 技術指標計算時陣列形狀不一致

---

## 問題分析

### 問題代碼位置

#### 1. `calculate_atr()` 方法（原代碼第 293-302 行）

**原始代碼：**
```python
@staticmethod
def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    high = np.array(high)
    low = np.array(low)
    close = np.array(close)
    
    tr = np.maximum(
        high - low,                           # Shape: (100,)
        np.maximum(
            np.abs(high - close[:-1]),        # Shape: (99,) ← 問題！
            np.abs(low - close[:-1])          # Shape: (99,) ← 問題！
        )
    )
    # ...
```

**問題原因：**
- `high - low` 的形狀是 `(100,)`
- `close[:-1]` 的形狀是 `(99,)`（去掉最後一個元素）
- 在 `np.maximum()` 中無法進行廣播，因為形狀不相容

#### 2. `calculate_adx()` 方法（原代碼第 269-282 行）

類似的形狀不一致問題。

---

## 解決方案

### 修復方法 1：`calculate_atr()` 方法

**新代碼：**
```python
@staticmethod
def calculate_atr(high, low, close, period=14):
    """Average True Range - FIXED: Proper array alignment"""
    high = np.array(high, dtype=float)
    low = np.array(low, dtype=float)
    close = np.array(close, dtype=float)
    
    # 確保所有陣列長度一致
    min_len = min(len(high), len(low), len(close))
    high = high[-min_len:]
    low = low[-min_len:]
    close = close[-min_len:]
    
    # 計算 True Range
    # TR = max(high - low, abs(high - previous close), abs(low - previous close))
    tr1 = high - low
    
    # 對前一個 close 進行填充（第一個元素用當前 close）
    close_prev = np.concatenate([[close[0]], close[:-1]])
    tr2 = np.abs(high - close_prev)           # Shape: (100,)
    tr3 = np.abs(low - close_prev)            # Shape: (100,)
    
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # 計算 ATR（TR 的 SMA）
    atr = np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)
    return float(atr)
```

**關鍵改進：**
1. 確保所有輸入陣列有相同長度
2. 使用 `np.concatenate()` 填充前一個 close，保持形狀一致
3. 所有運算的陣列形狀都是 `(100,)`

### 修復方法 2：`calculate_adx()` 方法

**新代碼：**
```python
@staticmethod
def calculate_adx(high, low, close, period=14):
    """Average Directional Index (simplified)"""
    high = np.array(high)
    low = np.array(low)
    close = np.array(close)
    
    # 確保所有陣列有相同的長度
    min_len = min(len(high), len(low), len(close))
    high = high[-min_len:]
    low = low[-min_len:]
    close = close[-min_len:]
    
    tr = np.maximum(high - low, np.abs(high - close[0]))  # 改用第一個 close
    atr = np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)
    
    adx = 50 + (high[-1] - low[-1]) / atr * 10 if atr > 0 else 50
    return float(np.clip(adx, 0, 100))
```

### 修復方法 3：其他改進

#### `calculate_rsi()` 方法
```python
@staticmethod
def calculate_rsi(prices, period=14):
    """Relative Strength Index"""
    prices = np.array(prices)
    deltas = np.diff(prices)  # Shape: (99,)
    
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # 改進：考慮陣列長度
    avg_gain = np.mean(gains[-period:]) if len(gains) >= period else np.mean(gains)
    avg_loss = np.mean(losses[-period:]) if len(losses) >= period else np.mean(losses)
    # ...
```

#### `calculate_volatility()` 方法
```python
@staticmethod
def calculate_volatility(prices, period=20):
    """Historical volatility"""
    prices = np.array(prices, dtype=float)
    if len(prices) < 2:
        return 0.0  # 防止空陣列
    
    returns = np.diff(prices) / prices[:-1]
    volatility = np.std(returns[-period:]) if len(returns) >= period else np.std(returns)
    return float(volatility)
```

#### `generate_price_data()` 函數
```python
def generate_price_data(current_price, count):
    """Generate simulated price data"""
    prices = [current_price]
    price = current_price
    
    for _ in range(count - 1):
        change = np.random.normal(0, 0.01)
        price = price * (1 + change)
        prices.insert(0, price)
    
    return prices[:count]  # 確保正確數量
```

#### `predict()` 端點
```python
@app.route('/api/v6/predict', methods=['POST'])
def predict():
    try:
        # ...
        # 確保所有陣列有相同長度
        prices = np.array(prices)
        if len(prices) != klines_count:
            prices = prices[-klines_count:]  # 取最後 klines_count 個
        
        # 計算技術指標 - 所有陣列長度相同
        indicators = {
            'RSI': TechnicalIndicators.calculate_rsi(prices),
            'MACD': TechnicalIndicators.calculate_macd(prices),
            'ADX': TechnicalIndicators.calculate_adx(prices, prices, prices),  # 傳入相同陣列
            'ATR': TechnicalIndicators.calculate_atr(prices, prices, prices),  # 傳入相同陣列
            'Volatility': TechnicalIndicators.calculate_volatility(prices)
        }
        # ...
```

---

## 修復總結

| 問題 | 原因 | 解決方案 |
|------|------|--------|
| `(100,)` vs `(99,)` 形狀不匹配 | `close[:-1]` 移除最後元素 | 使用 `np.concatenate()` 填充第一個元素 |
| 陣列長度不一致 | 資料預處理時未確保統一 | 在計算前驗證所有陣列長度 |
| `np.diff()` 導致形狀縮小 | 計算差值時長度 -1 | 在計算 RSI、波動率前進行檢查 |
| 無防守的除以零 | ATR 或其他指標為 0 | 添加 `if atr > 0` 檢查 |

---

## 測試方法

### 1. 啟動 API 伺服器
```bash
python app.py
```

### 2. 測試預測端點
```bash
curl -X POST http://localhost:8001/api/v6/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "timeframe": "1d",
    "klines": 100
  }'
```

### 3. 應該看到正常回應
```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1d",
  "klines_count": 100,
  "current_price": 42150.25,
  "predicted_price": 42650.75,
  "confidence": 0.68,
  "recommendation": "BUY",
  "entry_price": 42150.25,
  "stop_loss": 41327.25,
  "take_profit": 43483.77,
  "technical_indicators": {...},
  "volatility": {...},
  "model_predictions": {...},
  "timestamp": "2025-12-26T..."
}
```

---

## 提交信息

**Commit SHA:** `01e43a7612abefaff1c4e3f2af4ce2eb0f2349da`

**修改檔案：** `app.py`

**修改行數：**
- `calculate_atr()`: 第 247-268 行
- `calculate_adx()`: 第 225-242 行  
- `calculate_rsi()`: 第 212-224 行
- `calculate_volatility()`: 第 272-278 行
- `predict()` 端點: 第 376-393 行
- `generate_price_data()`: 第 449-460 行

---

## 相關文檔

- [NumPy Broadcasting Rules](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [Array Shape Management](https://numpy.org/doc/stable/user/basics.shapes.html)
- [Flask API 設計最佳實踐](https://flask.palletsprojects.com/)
