# 圖表改進指南 - CPB Crypto Predictor V6.1

## 改進概述

### 舊版圖表問題

❌ **原始圖表存在的問題：**
1. 歷史價格線和預測線連接不清楚
2. K 線標籤排列不合理（預測部分缺乏明確標記）
3. 缺乏移動平均線（MA7、MA21）的視覺支援
4. 圖例位置靠上，容易與資料重疊
5. 點標記不清晰，容易混淆
6. 預測線的起點與當前價格的連接邏輯不清
7. 整體色彩搭配不夠專業

### 新版圖表改進

✅ **V6.1 新版改進特性：**

#### 1. **清晰的預測線分離**
- 歷史線：實線（紫色漸變 #7c3aed → #a855f7）
- 預測線：虛線（青色 #06b6d4，明確區別）
- 預測線從當前價格直接延伸到預測價格

#### 2. **改進的技術指標顯示**
- MA(7)：綠色線（較高頻率移動平均）
- MA(21)：橙色線（較低頻率移動平均）
- 易於判斷趨勢方向

#### 3. **增強的圖例系統**
- 圖例位置：圖表上方居中
- 彩色方塊清晰標識各線條
- 實線與虛線視覺區分明顯

#### 4. **改進的資訊卡片**
- 當前價格
- 預測價格
- 變動方向（↑ UP / ↓ DOWN）
- 變動百分比
- 顏色編碼：上升藍色、下跌紅色

#### 5. **優化的互動體驗**
- Hover 工具提示（tooltip）
- 格式化的價格顯示（$X,XXX.XX）
- 平滑的線條渲染（tension: 0.4）
- 響應式設計

#### 6. **專業的設計風格**
- 深色主題（#1a1b26 背景）
- 玻璃態效果（backdrop blur）
- 漸變標題（#7c3aed → #06b6d4）
- 一致的間距和對齐

## 新增 API 端點

### 1. `/api/v6/chart` - 價格預測圖表

**請求方式：POST**

```bash
curl -X POST http://localhost:8001/api/v6/chart \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "timeframe": "1d",
    "klines": 100,
    "source": "binance"
  }'
```

**返回值：** HTML 頁面（帶嵌入的 Chart.js）

**包含內容：**
- 互動式K線圖
- 歷史價格線
- 預測價格線（虛線）
- MA(7) 和 MA(21)
- 實時價格資訊卡片

### 2. `/api/v6/indicators` - 技術指標儀表板

**請求方式：POST**

```bash
curl -X POST http://localhost:8001/api/v6/indicators \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "timeframe": "1d",
    "klines": 100
  }'
```

**返回值：** HTML 頁面（技術指標儀表板）

**顯示指標：**
- **RSI (14)** - 相對強弱指數（0-100 進度條）
- **MACD** - 移動平均線收斂發散（正負值）
- **ADX (14)** - 平均方向指標（趨勢強度）
- **ATR (14)** - 真實波幅
- **Volatility** - 歷史波動率

**指標解釋：**

#### RSI
```
≥ 70: 超買信號 - 潛在賣壓
30-70: 中性區域
≤ 30: 超賣信號 - 潛在買機
```

#### MACD
```
> 0: 正 MACD - 看漲信號
< 0: 負 MACD - 看跌信號
= 0: 中性
```

#### ADX
```
≥ 50: 非常強勢的趨勢
30-50: 強勢趨勢
20-30: 弱勢趨勢
< 20: 無趨勢 - 區間震盪
```

## 視覺元素詳解

### 顏色方案

| 元素 | 顏色 | 用途 |
|------|------|------|
| 主背景 | #1a1b26 | 深色背景 |
| 歷史線 | #a855f7 | 紫色漸變 |
| 預測線 | #06b6d4 | 青色（虛線） |
| MA(7) | #22c55e (透明度 0.6) | 綠色 |
| MA(21) | #f59e0b (透明度 0.6) | 橙色 |
| 上升箭頭 | #06b6d4 | 青色 |
| 下跌箭頭 | #ef4444 | 紅色 |

### 線條風格

```javascript
// 歷史價格線 - 實線，寬 3px
borderWidth: 3
borderDash: []
tension: 0.4  // 平滑曲線

// 預測線 - 虛線，寬 3px
borderWidth: 3
borderDash: [5, 5]  // 5px 線 + 5px 空隙
tension: 0.4

// 移動平均線 - 實線，寬 2px
borderWidth: 2
borderDash: []
tension: 0.4
```

### 點標記風格

```javascript
// 歷史數據點
pointRadius: 4              // 平時半徑
pointHoverRadius: 6         // Hover 時半徑
pointBackgroundColor: '#7c3aed'
pointBorderColor: '#fff'    // 白色邊框
pointBorderWidth: 2

// 預測點
pointRadius: 6              // 較大，突出重要性
pointHoverRadius: 8
pointBackgroundColor: '#0891b2'
```

## 使用案例

### 案例 1：在網頁中嵌入圖表

```html
<!DOCTYPE html>
<html>
<head>
    <title>Price Chart</title>
</head>
<body>
    <iframe src="http://localhost:8001/api/v6/chart" 
            style="width: 100%; height: 500px; border: none;"></iframe>
</body>
</html>
```

### 案例 2：通過 Python 獲取圖表 HTML

```python
import requests

payload = {
    'symbol': 'BTCUSDT',
    'timeframe': '1d',
    'klines': 100,
    'source': 'binance'
}

response = requests.post('http://localhost:8001/api/v6/chart', json=payload)

# 保存為 HTML 檔案
with open('price_chart.html', 'w') as f:
    f.write(response.text)

# 在瀏覽器中打開
import webbrowser
webbrowser.open('price_chart.html')
```

### 案例 3：獲取技術指標儀表板

```python
response = requests.post('http://localhost:8001/api/v6/indicators', json=payload)

with open('indicators_dashboard.html', 'w') as f:
    f.write(response.text)
```

## 改進前後對比

### 舊版缺陷

```
圖例位置：上方靠右（易被資料覆蓋）
預測線：紫色實線（與歷史線無區別）
線條風格：無張力（看起來生硬）
點標記：小且難以識別
MA 線：無顯示
```

### 新版改進

```
圖例位置：上方居中（清晰可見）
預測線：青色虛線（一目瞭然）
線條風格：tension: 0.4（平滑自然）
點標記：大且清晰（易於交互）
MA 線：綠色和橙色清晰顯示
```

## 技術棧

- **前端圖表庫**: Chart.js 4.4.0
- **樣式系統**: 原生 CSS3（漸變、模糊效果、Flexbox）
- **響應式設計**: 媒體查詢 + 彈性布局
- **交互增強**: Tooltip + Hover 效果
- **色彩空間**: RGB + RGBA (透明度支援)

## 性能優化

1. **CDN 加載**: Chart.js 從 CDN 加載（減少伺服器負擔）
2. **緩存策略**: 圖表複用市場資料快取
3. **渲染優化**: 使用 `maintainAspectRatio: true` 避免重排
4. **數據處理**: Python 側計算 MA，減少前端計算

## 常見問題

### Q1: 預測線為什麼從當前價格開始？

A: 為了清晰顯示當前價格和預測價格的差異，預測線從當前 K 線的末端（當前價格）延伸到下一個時間點（預測價格）。

### Q2: 如何修改線條顏色？

A: 在 `visualization.py` 中修改 `borderColor` 屬性：
```python
'borderColor': '#a855f7',  # 改為你想要的顏色
```

### Q3: 如何增加更多技術指標？

A: 在 `ChartGenerator.generate_price_chart()` 中添加新的 dataset：
```python
{
    'label': 'New Indicator',
    'data': new_indicator_data,
    'borderColor': '#color',
    ...
}
```

### Q4: 圖表無法加載怎麼辦？

A: 1. 檢查伺服器是否運行
   2. 確認 Binance/yfinance 資料可用
   3. 查看瀏覽器控制台的錯誤訊息
   4. 檢查 CORS 配置

## 下一步增強計劃

1. **實時更新** - WebSocket 支援自動刷新
2. **自定義指標** - 允許用戶選擇要顯示的指標
3. **導出功能** - 支援 PNG/SVG 導出圖表
4. **對比分析** - 多個交易對同時對比
5. **模式識別** - 標記 K 線形態（頭肩頂、雙底等）
6. **告警標記** - 顯示交易信號觸發點

## 檔案清單

- `visualization.py` - 圖表生成核心模組（新增）
- `app.py` - 更新的 API 伺服器（v6.1，新增 `/chart` 和 `/indicators` 端點）
- `CHART_IMPROVEMENTS.md` - 本檔案

## 使用步驟

1. **啟動伺服器**
   ```bash
   python app.py
   ```

2. **訪問圖表**
   ```bash
   # 方法 1：直接訪問
   curl -X POST http://localhost:8001/api/v6/chart \
     -H "Content-Type: application/json" \
     -d '{"symbol": "BTCUSDT", "timeframe": "1d", "klines": 100}' \
     > chart.html
   
   # 方法 2：在瀏覽器中打開
   # http://localhost:8001/api/v6/chart (需要 POST body)
   ```

3. **查看指標儀表板**
   ```bash
   curl -X POST http://localhost:8001/api/v6/indicators \
     -H "Content-Type: application/json" \
     -d '{"symbol": "BTCUSDT", "timeframe": "1d", "klines": 100}' \
     > indicators.html
   ```

## 版本資訊

- **版本**: V6.1
- **更新日期**: 2025-12-26
- **狀態**: 正式發布
- **改進**: 圖表、指標儀表板、API 端點

---

**完美的圖表讓交易決策更明確。享受改進後的視覺化體驗！**
