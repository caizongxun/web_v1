# CPB Crypto Predictor Web - V6 Model

## 概述 (Overview)

CPB Crypto Predictor Web 是一個專業的加密貨幣價格預測應用，採用 V6 混合模型，整合了 LSTM、GRU 和 XGBoost 三種深度學習和機器學習技術。

**CPB Crypto Predictor Web** is a professional cryptocurrency price prediction application using the V6 hybrid model, integrating LSTM, GRU, and XGBoost technologies.

### 核心特性 (Key Features)

- 支持 19 種加密貨幣 (19 Supported Cryptocurrencies)
- 3 種時間框架 (3 Timeframes: 1d, 1h, 15m)
- 混合模型預測 (Hybrid Model Prediction)
- 風險管理工具 (Risk Management Tools)
- 實時技術指標 (Real-time Technical Indicators)
- 專業級交易信號 (Professional Trading Signals)

---

## V6 模型架構 (V6 Model Architecture)

### 混合模型組合 (Hybrid Model Combination)

```
┌─────────────────┐
│  Input: Prices  │
└────────┬────────┘
         │
    ┌────┴────┐
    │          │
┌───▼──┐   ┌──▼────┐   ┌──────────┐
│ LSTM │   │ GRU   │   │ XGBoost  │
│ (50%)│   │ (30%) │   │  (20%)   │
└───┬──┘   └──┬────┘   └──┬───────┘
    │         │           │
    └────┬────┴──────┬────┘
         │           │
    ┌────▼───────────▼────┐
    │  Weighted Ensemble  │
    │  Final Prediction   │
    └─────────────────────┘
```

### 模型詳細規格 (Detailed Specifications)

#### LSTM (50% 權重)
- **層數**: 3 層
- **隱層大小**: 128 → 64 → 32
- **Dropout**: 0.2
- **用途**: 捕捉長期時序依賴

#### GRU (30% 權重)
- **層數**: 2 層
- **隱層大小**: 96 → 48
- **Dropout**: 0.15
- **用途**: 捕捉中期趨勢變化

#### XGBoost (20% 權重)
- **樹深**: 6
- **學習率**: 0.1
- **樹數量**: 200
- **用途**: 捕捉非線性特徵關係

### 特徵工程 (Feature Engineering)

#### 動量指標 (Momentum Indicators)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator

#### 趨勢指標 (Trend Indicators)
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- ADX (Average Directional Index)

#### 波動率指標 (Volatility Indicators)
- ATR (Average True Range)
- Bollinger Bands
- Historical Volatility

#### 成交量指標 (Volume Indicators)
- OBV (On-Balance Volume)
- VWAP (Volume Weighted Average Price)
- Volume Ratio

#### 其他特徵 (Other Features)
- 日期特徵 (Date Features)
- Log Returns
- BTC相關係數 (Bitcoin Correlation)

---

## 支持的加密貨幣 (Supported Cryptocurrencies)

### 完整支持 (Full Support - 15 種)

所有時間框架可用: 1d (日線)、1h (小時線)、15m (15分鐘線)

All timeframes available: 1d, 1h, 15m

```
BTC   - Bitcoin
ETH   - Ethereum
BNB   - Binance Coin
SOL   - Solana
ADA   - Cardano
DOGE  - Dogecoin
AVAX  - Avalanche
DOT   - Polkadot
LTC   - Litecoin
LINK  - Chainlink
ATOM  - Cosmos
NEAR  - NEAR Protocol
ICP   - Internet Computer
CRO   - Crypto.com
HBAR  - Hedera
```

### 部分支持 (Partial Support - 4 種)

限制時間框架 (Limited Timeframes)

```
VET   - VeChain        (1d, 15m - 缺 1h)
MATIC - Polygon        (1d - 僅日線)
FTM   - Fantom         (1d - 僅日線)
UNI   - Uniswap        (1d - 僅日線)
```

---

## 時間框架與精度 (Timeframes & Accuracy)

### 1d (日線)
- **用途**: 中長期交易
- **K 線範圍**: 30-100 根
- **預期精度**: ~72%

### 1h (小時線)
- **用途**: 日內交易
- **K 線範圍**: 24-168 根
- **預期精度**: ~68%

### 15m (15分鐘線)
- **用途**: 超短期交易
- **K 線範圍**: 96-672 根
- **預期精度**: ~62%

---

## 風險管理層 (Risk Management Layer)

### 進場點 (Entry Price)

```
IF 預測價格 > 當前價格:
    Entry Price = 當前價格 × 1.005 (加 0.5% 溢價)
ELSE:
    Entry Price = 預測價格
```

### 止損 (Stop Loss)

```
Stop Loss = Entry Price × (1 - ATR/Entry Price × 1.5)

基於 ATR (真實波幅) 自動調整風險
```

### 獲利目標 (Take Profit)

```
Risk = Entry Price - Stop Loss
Take Profit = Entry Price + Risk × 2.0

風險回報比: 1:2
```

### 信心度計算 (Confidence Calculation)

```
Confidence = Model Consensus + Technical Confirmation

• 模型預測標準差越小 → 信心度越高
• 三個技術指標同向 (RSI, MACD, Trend) → 加分
• 數值範圍: 0.0 ~ 1.0 (0-100%)
```

### 交易信號 (Trading Signals)

#### BUY 信號
```
IF 預測價格 > 當前價格
AND 信心度 > 0.5
AND RSI < 70
AND MACD 金叉或正值
    → BUY
```

#### SELL 信號
```
IF 預測價格 < 當前價格
AND 信心度 > 0.5
AND RSI > 30
AND MACD 死叉或負值
    → SELL
```

#### HOLD 信號
```
IF 信心度 < 0.5
OR 技術面無明確方向
    → HOLD
```

### 波動率評估 (Volatility Assessment)

```
Current:  最近 20 根 K 線的歷史波動率
Predicted: 模型預測的下期波動率

Level:
  低   (Low)    < 0.5%
  中   (Medium) 0.5% - 1.5%
  高   (High)   > 1.5%
```

---

## 快速開始 (Quick Start)

### 前置要求 (Prerequisites)

```bash
# Python 3.8+
python --version

# pip
pip --version
```

### 安裝 (Installation)

#### 1. 後端安裝

```bash
# 安裝依賴
pip install -r requirements.txt

# 啟動 API 服務器 (Port 8001)
python app.py
```

#### 2. 前端啟動

```bash
# 使用 Python http.server (推薦)
python -m http.server 8000

# 或使用 Node.js http-server
npx http-server -p 8000
```

#### 3. 訪問應用

```
前端: http://localhost:8000
API: http://localhost:8001
```

---

## API 規範 (API Specification)

### 預測端點 (Prediction Endpoint)

#### 請求 (Request)

```bash
curl -X POST http://localhost:8001/api/v6/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "timeframe": "1d",
    "klines": 100
  }'
```

#### 響應 (Response)

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "1d",
  "klines_count": 100,
  "current_price": 45250.50,
  "predicted_price": 46100.75,
  "confidence": 0.75,
  "recommendation": "BUY",
  "entry_price": 45456.76,
  "stop_loss": 44500.50,
  "take_profit": 46412.76,
  "technical_indicators": {
    "RSI": 62.5,
    "MACD": 125.30,
    "ADX": 45.2,
    "ATR": 450.5
  },
  "volatility": {
    "current": 0.0125,
    "predicted": 0.0135,
    "level": "Medium"
  },
  "model_predictions": {
    "LSTM": 46050.25,
    "GRU": 46120.50,
    "XGBoost": 46150.00
  },
  "model_config": {
    "version": "V6",
    "lstm_weight": 0.5,
    "gru_weight": 0.3,
    "xgboost_weight": 0.2
  },
  "accuracy": 0.72,
  "timestamp": "2025-12-26T15:40:00.000Z"
}
```

### 支持符號端點 (Symbols Endpoint)

```bash
GET http://localhost:8001/api/v6/symbols
```

### 模型配置端點 (Config Endpoint)

```bash
GET http://localhost:8001/api/v6/config
```

### 健康檢查端點 (Health Check)

```bash
GET http://localhost:8001/api/v6/health
```

---

## 項目結構 (Project Structure)

```
web_v1/
├── index.html          # 前端主頁面
├── app.js              # 前端應用邏輯
├── app.py              # 後端 Flask API
├── requirements.txt    # Python 依賴
├── README.md          # 文檔
├── .env               # 環境變數 (可選)
└── models/            # 模型文件夾 (生產用)
    ├── lstm_model.h5
    ├── gru_model.h5
    └── xgboost_model.pkl
```

---

## 文件說明 (File Description)

### index.html
- 響應式前端界面
- 專業級 UI/UX 設計
- Chart.js 圖表集成
- 完整的表單驗證

### app.js
- API 交互邏輯
- 表單處理
- 結果展示
- 圖表渲染

### app.py
- Flask Web 框架
- V6 模型實現
- 技術指標計算
- 風險管理邏輯

---

## 生產部署 (Production Deployment)

### 後端部署

```bash
# 使用 Gunicorn
gunicorn -w 4 -b 0.0.0.0:8001 app:app

# 或使用 Docker
docker build -t cpb-v6 .
docker run -p 8001:8001 cpb-v6
```

### 前端部署

```bash
# Nginx 配置
server {
    listen 8000;
    location / {
        root /var/www/cpb-v6;
        try_files $uri /index.html;
    }
}
```

---

## 免責聲明 (Disclaimer)

⚠️ **重要提示**: 

本預測工具僅用於教育和分析目的。加密貨幣市場高度波動且不可預測。過去表現不保證未來結果。投資前請進行充分研究並諮詢財務顧問。

**Important Notice**: 

This prediction tool is for educational and analytical purposes only. Cryptocurrency markets are highly volatile and unpredictable. Past performance does not guarantee future results. Always conduct your own research and consult with financial advisors before making investment decisions.

---

## 授權 (License)

MIT License

---

## 聯繫方式 (Contact)

- GitHub: [caizongxun/web_v1](https://github.com/caizongxun/web_v1)
- 版本: V6.0
- 最後更新: 2025-12-26

---

## 更新日誌 (Changelog)

### V6.0 (2025-12-26)
- 完整的混合模型實現
- 19 種加密貨幣支持
- 專業級風險管理
- 響應式前端界面
- 完整 API 規範

---

**Made with by CPB Team | 專業加密貨幣預測平台**
