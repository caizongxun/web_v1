# CPB Crypto Predictor Web V6 - 快速開始指南

## 系統要求 (System Requirements)

- Python 3.8+
- pip 或 conda
- Node.js (可選，用於前端開發)
- Docker (可選，用於容器化部署)

---

## 開發環境設置 (Development Setup)

### 方法 1: 本地開發 (Local Development)

#### 步驟 1: 克隆倉庫

```bash
git clone https://github.com/caizongxun/web_v1.git
cd web_v1
```

#### 步驟 2: 設置後端環境

```bash
# 創建虛擬環境
python -m venv venv

# 激活虛擬環境
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 安裝依賴
pip install -r requirements.txt

# 複製環境配置
cp .env.example .env
```

#### 步驟 3: 啟動後端 API 服務

```bash
# 開發模式
python app.py

# 或使用 Gunicorn (生產模式)
gunicorn -w 4 -b 0.0.0.0:8001 app:app
```

✅ 後端運行在: http://localhost:8001

#### 步驟 4: 啟動前端服務

打開新的終端窗口:

```bash
cd web_v1

# 方法 A: 使用 Python 內建伺服器
python -m http.server 8000

# 方法 B: 使用 Node.js http-server (需先安裝)
npx http-server -p 8000

# 方法 C: 使用任何其他 Web 伺服器
```

✅ 前端運行在: http://localhost:8000

#### 步驟 5: 訪問應用

在瀏覽器中打開: **http://localhost:8000**

---

### 方法 2: Docker 容器部署 (Docker Deployment)

#### 單容器啟動

```bash
# 構建鏡像
docker build -t cpb-v6 .

# 運行容器
docker run -p 8001:8001 cpb-v6
```

#### Docker Compose 完整棧

```bash
# 啟動所有服務
docker-compose up -d

# 查看日誌
docker-compose logs -f

# 停止服務
docker-compose down
```

✅ 訪問:
- 前端: http://localhost
- API: http://localhost:8001
- Nginx: http://localhost:80

---

## 測試預測端點 (Testing Prediction Endpoint)

### 使用 cURL

```bash
curl -X POST http://localhost:8001/api/v6/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "timeframe": "1d",
    "klines": 100
  }'
```

### 使用 Python requests

```python
import requests
import json

url = 'http://localhost:8001/api/v6/predict'
data = {
    'symbol': 'BTCUSDT',
    'timeframe': '1d',
    'klines': 100
}

response = requests.post(url, json=data)
result = response.json()
print(json.dumps(result, indent=2))
```

### 使用 JavaScript Fetch

```javascript
const prediction = await fetch('http://localhost:8001/api/v6/predict', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        symbol: 'BTCUSDT',
        timeframe: '1d',
        klines: 100
    })
});

const result = await prediction.json();
console.log(result);
```

---

## API 健康檢查 (Health Check)

```bash
curl http://localhost:8001/api/v6/health
```

預期響應:
```json
{
  "status": "healthy",
  "version": "V6",
  "timestamp": "2025-12-26T15:40:00.000Z"
}
```

---

## 獲取支持的符號 (Get Supported Symbols)

```bash
curl http://localhost:8001/api/v6/symbols
```

---

## 常見問題 (Troubleshooting)

### 問題 1: 連接被拒絕 (Connection Refused)

**症狀**: `Error: connect ECONNREFUSED 127.0.0.1:8001`

**解決方案**:
```bash
# 確保後端服務運行
python app.py

# 檢查端口
lsof -i :8001  # macOS/Linux
netstat -ano | findstr :8001  # Windows
```

### 問題 2: CORS 錯誤

**症狀**: `Cross-Origin Request Blocked`

**解決方案**:
- CORS 已在 app.py 中配置
- 確保使用 http://localhost:8000 而不是 file://

### 問題 3: Python 依賴缺失

**症狀**: `ModuleNotFoundError: No module named 'flask'`

**解決方案**:
```bash
pip install -r requirements.txt
```

### 問題 4: 端口已被占用

**症狀**: `Address already in use`

**解決方案**:
```bash
# 更改端口
python app.py --port 8002

# 或殺死占用端口的進程
lsof -ti :8001 | xargs kill -9  # macOS/Linux
```

---

## 開發工作流程 (Development Workflow)

### 前端開發

1. 編輯 `index.html` 修改 UI
2. 編輯 `app.js` 修改邏輯
3. 在瀏覽器中刷新頁面查看更改
4. 打開開發者工具 (F12) 查看控制台

### 後端開發

1. 編輯 `app.py` 修改 API 邏輯
2. 當使用 Flask debug 模式時，服務器會自動重新加載
3. 測試 API 端點
4. 檢查日誌輸出

### 添加新的模型

```python
# 在 app.py 中添加新模型類
class V7Model:
    def predict(self, prices, volumes, indicators):
        # 實現預測邏輯
        return predicted_price, confidence
```

---

## 性能優化 (Performance Optimization)

### 後端優化

```bash
# 使用更多的 Gunicorn workers
gunicorn -w 8 -b 0.0.0.0:8001 app:app

# 使用 uvicorn (如果遷移到 FastAPI)
uvicorn app:app --host 0.0.0.0 --port 8001 --workers 8
```

### 前端優化

```html
<!-- 最小化 JavaScript -->
<!-- 使用 CDN 加速 Chart.js -->
<!-- 啟用瀏覽器緩存 -->
```

---

## 生產部署檢查清單 (Production Deployment Checklist)

- [ ] 禁用 Flask debug 模式
- [ ] 設置環境變數 `FLASK_ENV=production`
- [ ] 配置 CORS 為特定域名
- [ ] 添加 SSL/TLS 證書
- [ ] 設置日誌輪轉 (Log Rotation)
- [ ] 配置監控和告警
- [ ] 備份數據庫
- [ ] 設置速率限制
- [ ] 實施身份驗證 (如需要)
- [ ] 性能測試和優化

---

## 進一步閱讀 (Further Reading)

- [完整文檔](README.md)
- [Flask 官方文檔](https://flask.palletsprojects.com/)
- [Chart.js 文檔](https://www.chartjs.org/docs/latest/)
- [Docker 文檔](https://docs.docker.com/)

---

**需要幫助?** 查看 README.md 或在 GitHub 上提交 Issue。
