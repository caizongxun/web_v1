# 實時市場資料集成指南

## 概述

CPB Crypto Predictor V6.1 現已支援實時市場資料整合，支援以下資料來源：

- **Binance API** - 專業級加密貨幣交易所資料
- **yfinance** - Yahoo Finance 財務資料

## 新功能特性

### 1. 雙資料源支援

| 特性 | Binance | yfinance |
|------|---------|----------|
| **資料來源** | 真實交易所資料 | Yahoo Finance |
| **更新頻率** | 實時 | 日更新 |
| **交易對** | 1000+ 加密貨幣 | 主流資產 |
| **時間框架** | 1m - 1M | 1m - 1mo |
| **API 限制** | 1200 req/min | 無硬性限制 |
| **最佳用途** | 日內交易、短期 | 長期分析 |

### 2. 智能快取系統

- 自動 5 分鐘快取機制
- 防止 API 速率限制
- 可配置的快取策略

### 3. 資料驗證

- 完整的 OHLCV 資料驗證
- NaN 值檢查
- 邊界條件驗證
- 自動資料清理

## 安裝與設置

### 步驟 1：更新依賴

```bash
pip install -r requirements.txt
```

新增的依賴：
```
yfinance==0.2.32
```

### 步驟 2：環境變數配置（可選）

在 `.env` 檔案中設置：

```env
# 預設資料來源 (binance 或 yfinance)
DATA_SOURCE=binance

# 啟用快取 (true 或 false)
CACHE_ENABLED=true

# Binance API 密鑰（可選）
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
```

### 步驟 3：啟動伺服器

```bash
python app.py
```

日誌輸出應顯示：
```
2025-12-26 16:31:07 - __main__ - INFO - Starting CPB Crypto Predictor V6.1 API Server
2025-12-26 16:31:07 - __main__ - INFO - Default Data Source: binance
2025-12-26 16:31:07 - __main__ - INFO - Cache Enabled: True
```

## API 使用方式

### 基本用法（使用預設資料源）

```bash
curl -X POST http://localhost:8001/api/v6/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "timeframe": "1d",
    "klines": 100
  }'
```

### 指定資料來源

#### 使用 Binance 資料
```bash
curl -X POST http://localhost:8001/api/v6/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "timeframe": "1d",
    "klines": 100,
    "source": "binance"
  }'
```

#### 使用 yfinance 資料
```bash
curl -X POST http://localhost:8001/api/v6/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "timeframe": "1d",
    "klines": 100,
    "source": "yfinance"
  }'
```

### 禁用快取

```bash
curl -X POST http://localhost:8001/api/v6/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "timeframe": "1d",
    "klines": 100,
    "cache": false
  }'
```

## API 響應格式

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1d",
  "klines_count": 100,
  "data_source": "binance",
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
  "model_predictions": {
    "LSTM": 42500.50,
    "GRU": 42700.25,
    "XGBoost": 42800.00
  },
  "timestamp": "2025-12-26T16:31:07.123456"
}
```

**新增欄位**
- `data_source`: 使用的資料來源（可能包括 "(cached)" 標記）

## 支援的交易對

### Binance 完全支援
```
BTC, ETH, BNB, SOL, ADA, DOGE, AVAX, DOT, LTC,
LINK, ATOM, NEAR, ICP, CRO, HBAR
```

所有交易對均以 USDT 配對（例如 BTCUSDT）

### yfinance 常用格式
```
BTC-USD, ETH-USD, BNB-USD, SOL-USD, ADA-USD
DOGE-USD, AVAX-USD, DOT-USD, LTC-USD, LINK-USD
```

## 時間框架支援

### Binance
```
1m, 5m, 15m, 1h, 4h, 1d, 1w, 1M
```

### yfinance
```
1m, 5m, 15m, 1h, 4h, 1d, 1w (1wk), 1M (1mo)
```

## 資料驗證邏輯

系統會自動驗證資料的完整性：

```python
# 檢查項目
1. 最少 K 線數量 (min_klines)
2. NaN 值檢查
3. 正價格檢查
4. OHLC 邏輯驗證 (High >= Close >= Low)
5. 成交量驗證
```

## 快取配置

### 快取鍵格式
```
{source}:{symbol}:{timeframe}:{klines_count}
```

### 快取持續時間
```
預設: 300 秒 (5 分鐘)
```

### 快取方法
```
內存快取 (in-memory)
應用重啟時清空
```

## 性能優化

### 建議配置

**用於即時交易**
```json
{
  "source": "binance",
  "cache": true,
  "timeframe": "1h",
  "klines": 100
}
```

**用於長期分析**
```json
{
  "source": "yfinance",
  "cache": true,
  "timeframe": "1d",
  "klines": 1000
}
```

**用於回測**
```json
{
  "source": "yfinance",
  "cache": false,
  "timeframe": "1d",
  "klines": 500
}
```

## 常見問題解決

### Q1: 出現 "Unsupported cryptocurrency" 錯誤

**原因**: 交易對格式不正確或不支援

**解決**:
```bash
# 正確格式 (Binance)
"symbol": "BTCUSDT"   # ✓
"symbol": "BTC"       # ✗ 缺少 USDT

# 正確格式 (yfinance)
"symbol": "BTC-USD"   # ✓ 需要轉換
```

### Q2: 收到 "Invalid market data received" 錯誤

**原因**: API 返回無效或不完整的資料

**解決**:
1. 檢查網路連接
2. 確認 API 可用性
3. 嘗試禁用快取 `"cache": false`
4. 查看伺服器日誌取得詳細錯誤

### Q3: API 響應很慢

**原因**: 網路延遲或 API 伺服器繁忙

**解決**:
1. 啟用快取（預設啟用）
2. 減少 K 線數量
3. 使用 yfinance（更穩定但更新較慢）
4. 增加 API 超時時間（應用內配置）

### Q4: 如何批次測試多個交易對

**範例腳本**:
```python
import requests
import json

symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']

for symbol in symbols:
    payload = {
        'symbol': symbol,
        'timeframe': '1d',
        'klines': 100,
        'source': 'binance'
    }
    
    response = requests.post(
        'http://localhost:8001/api/v6/predict',
        json=payload
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"{symbol}: {data['recommendation']} (Confidence: {data['confidence']:.2%})")
    else:
        print(f"{symbol}: Error - {response.json()['error']}")
```

## 監控與日誌

### 重要日誌訊息

```
# 成功的資料取得
2025-12-26 16:31:10 - data_fetcher - INFO - Fetched 100 candles from Binance for BTCUSDT 1d

# 快取命中
2025-12-26 16:31:15 - data_fetcher - INFO - Cache hit: binance:BTCUSDT:1d:100

# 預測生成
2025-12-26 16:31:12 - __main__ - INFO - Prediction generated: BTCUSDT 1d from binance

# 驗證通過
2025-12-26 16:31:11 - data_fetcher - INFO - Data validation passed
```

## 版本歷史

### V6.1.0 (2025-12-26)
- 新增 Binance API 整合
- 新增 yfinance 支援
- 實現智能快取系統
- 完整的資料驗證框架
- 支援多資料源切換

## 相關檔案

- `data_fetcher.py` - 資料取得模組
- `app.py` - 主應用程式（V6.1）
- `requirements.txt` - 依賴列表
- `test_shape_fix.py` - 測試腳本

## 技術支援

如有問題，請檢查：
1. 伺服器日誌輸出
2. API 端點的健康狀態 `/api/v6/health`
3. 支援的加密貨幣列表 `/api/v6/symbols`
4. 模型配置 `/api/v6/config`

---

**最後更新**: 2025-12-26
**版本**: V6.1
**狀態**: 正式發佈
