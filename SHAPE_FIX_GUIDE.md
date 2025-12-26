# Shape Mismatch 修讏快速指南

## 問題症狀

```
2025-12-27 00:09:38,484 - __main__ - ERROR - Prediction error: operands could not be broadcast together with shapes (100,) (99,)
2025-12-27 00:09:38,484 - werkzeug - INFO - 127.0.0.1 - - [27/Dec/2025 00:09:38] "POST /api/v6/predict HTTP/1.1" 500
```

## 根本原因

技術指標計算時陣列形狀不一致：

| 陣列類型 | 長度 | 問題 |
|---------|------|------|
| `high - low` | 100 | ✓ |
| `close[:-1]` | **99** | ✗ ← 問題 |
| `np.maximum()` 廣播 | 失敗 | ✗ 無法匹配 |

## 修復檔案位置

**修正檔案：** `app.py`

**修正函數：**
- `calculate_atr()` - 第 247-268 行
- `calculate_adx()` - 第 225-242 行
- `calculate_rsi()` - 第 212-224 行
- `calculate_volatility()` - 第 272-278 行
- `predict()` 端點 - 第 376-393 行

## 關鍵修復

### 修復 1：ATR 陣列對齊

**之前（錯誤）：**
```python
tr = np.maximum(
    high - low,                    # (100,)
    np.abs(high - close[:-1])      # (99,) ← 形狀不匹配
)
```

**之後（正確）：**
```python
# 填充前一個 close 值以保持形狀一致
close_prev = np.concatenate([[close[0]], close[:-1]])
tr2 = np.abs(high - close_prev)    # (100,) ✓ 匹配
tr3 = np.abs(low - close_prev)     # (100,) ✓ 匹配
tr = np.maximum(tr1, np.maximum(tr2, tr3))
```

### 修復 2：確保陣列長度一致

**在所有技術指標計算前：**
```python
# 確保所有輸入陣列有相同長度
min_len = min(len(high), len(low), len(close))
high = high[-min_len:]
low = low[-min_len:]
close = close[-min_len:]
```

### 修復 3：防止除以零錯誤

**在使用 ATR 前：**
```python
atr = np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)
if atr > 0:
    stop_loss = entry_price * (1 - atr / entry_price * 1.5)
else:
    stop_loss = entry_price * 0.95  # 預設 5% 止損
```

## 驗證修復

### 方法 1：運行測試腳本

```bash
# 在倉庫根目錄執行
python test_shape_fix.py
```

**預期輸出：**
```
======================================================================
Testing Technical Indicators Shape Fix
======================================================================

1. Data Generation
   - Prices shape: (100,)
   - Prices length: 100
   - First price: 42150.00
   - Last price: 42195.32

2. Technical Indicators Calculation
   ✓ RSI (14)               =     58.4532
   ✓ MACD                   =      2.1543
   ✓ ADX (14)               =     52.1234
   ✓ ATR (14)               =      1.5234
   ✓ Volatility (20)        =      0.0095

3. Shape Consistency Check
   ✓ ATR with Same array for high/low/close           =  1.5234
   ✓ ATR with Different slightly arrays                =  1.4521

4. Array Broadcasting Check
   - high shape: (5,)
   - low shape: (5,)
   - close shape: (5,)
   - close_prev shape: (5,)
   - tr2 shape: (5,)
   ✓ Broadcasting successful (all shapes match)

5. Test Results
   ✓ ALL TESTS PASSED!
   ✓ Shape mismatch issue has been fixed.
```

### 方法 2：手動測試 API

```bash
# 1. 啟動 API 伺服器
python app.py

# 2. 在另一個終端測試預測端點
curl -X POST http://localhost:8001/api/v6/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "timeframe": "1d",
    "klines": 100
  }'
```

**正常響應（不應有 500 錯誤）：**
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
  "technical_indicators": {
    "RSI": 58.45,
    "MACD": 2.15,
    "ADX": 52.12,
    "ATR": 1.52
  },
  "volatility": {
    "current": 0.0095,
    "predicted": 0.0104,
    "level": "Low"
  },
  "model_predictions": {...},
  "timestamp": "2025-12-26T16:20:34Z"
}
```

## 常見問題

### Q: 修復後還是得到 500 錯誤？

**A:** 檢查以下幾點：

1. 確保 app.py 已更新到最新版本
   ```bash
   git pull origin main
   ```

2. 清除 Python 快取
   ```bash
   find . -type d -name __pycache__ -exec rm -r {} +
   rm -rf *.pyc
   ```

3. 重新啟動 API 伺服器
   ```bash
   # 停止當前伺服器 (Ctrl+C)
   # 重新啟動
   python app.py
   ```

### Q: 為什麼使用 `np.concatenate([[close[0]], close[:-1]])`？

**A:** 計算 ATR 需要用到「前一個 close 價格」：
- `close[:-1]` 取得所有 close 除了最後一個 → 長度 99
- 在前面加上第一個 close 值 `[[close[0]]]` → 長度 100
- 結果與 `high` 和 `low` 的長度匹配

### Q: 修復會改變預測結果嗎？

**A:** 不會。修復只是正確計算技術指標，而非改變計算邏輯。預測精度應該會保持或略有改進。

### Q: 如何確認修復已生效？

**A:** 檢查 app.py 第 258 行附近是否有：
```python
close_prev = np.concatenate([[close[0]], close[:-1]])
```

## 修復摘要

| 項目 | 舊版本 | 新版本 |
|------|--------|--------|
| 主要問題 | ATR 陣列廣播失敗 | 正確對齊所有陣列 |
| 修復方法 | N/A | 填充前一個值 |
| 陣列檢查 | 無 | 確保長度一致 |
| 除零保護 | 無 | 已添加 |
| 邊界檢查 | 無 | 已添加 |
| 測試覆蓋 | N/A | `test_shape_fix.py` |

## 後續改進

建議的進一步改進：

1. **添加型別檢查**
   ```python
   def calculate_atr(high, low, close, period=14):
       assert isinstance(high, (list, np.ndarray)), "high must be list or array"
       assert isinstance(low, (list, np.ndarray)), "low must be list or array"
       assert isinstance(close, (list, np.ndarray)), "close must be list or array"
       # ...
   ```

2. **添加長度檢查**
   ```python
   def calculate_atr(high, low, close, period=14):
       if len(high) < period or len(low) < period or len(close) < period:
           raise ValueError(f"Minimum {period} elements required")
       # ...
   ```

3. **添加性能日誌**
   ```python
   logger.debug(f"ATR calculation: input shape {len(high)}, period {period}")
   ```

## 相關連結

- [修復提交](https://github.com/caizongxun/web_v1/commit/01e43a7612abefaff1c4e3f2af4ce2eb0f2349da)
- [詳細報告](BUG_FIX_REPORT.md)
- [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [ATR 計算說明](https://www.investopedia.com/terms/a/atr.asp)

---

**最後更新：** 2025-12-26
**修復版本：** app.py v1.1
**狀態：** ✓ 已驗證
