// API Configuration
const API_BASE_URL = 'http://localhost:8001';
const API_ENDPOINTS = {
    predict: '/api/v6/predict',
    candles: '/api/v6/candles',
    technical: '/api/v6/technical',
    supported_symbols: '/api/v6/symbols'
};

// Supported cryptocurrencies with timeframe constraints
const CRYPTO_CONFIG = {
    'BTC': { fullSupport: true, timeframes: ['1d', '1h', '15m'] },
    'ETH': { fullSupport: true, timeframes: ['1d', '1h', '15m'] },
    'BNB': { fullSupport: true, timeframes: ['1d', '1h', '15m'] },
    'SOL': { fullSupport: true, timeframes: ['1d', '1h', '15m'] },
    'ADA': { fullSupport: true, timeframes: ['1d', '1h', '15m'] },
    'DOGE': { fullSupport: true, timeframes: ['1d', '1h', '15m'] },
    'AVAX': { fullSupport: true, timeframes: ['1d', '1h', '15m'] },
    'DOT': { fullSupport: true, timeframes: ['1d', '1h', '15m'] },
    'LTC': { fullSupport: true, timeframes: ['1d', '1h', '15m'] },
    'LINK': { fullSupport: true, timeframes: ['1d', '1h', '15m'] },
    'ATOM': { fullSupport: true, timeframes: ['1d', '1h', '15m'] },
    'NEAR': { fullSupport: true, timeframes: ['1d', '1h', '15m'] },
    'ICP': { fullSupport: true, timeframes: ['1d', '1h', '15m'] },
    'CRO': { fullSupport: true, timeframes: ['1d', '1h', '15m'] },
    'HBAR': { fullSupport: true, timeframes: ['1d', '1h', '15m'] },
    'VET': { fullSupport: false, timeframes: ['1d', '15m'] },
    'MATIC': { fullSupport: false, timeframes: ['1d'] },
    'FTM': { fullSupport: false, timeframes: ['1d'] },
    'UNI': { fullSupport: false, timeframes: ['1d'] }
};

// Accuracy levels by timeframe
const ACCURACY_LEVELS = {
    '1d': 0.72,
    '1h': 0.68,
    '15m': 0.62
};

// Model weights
const MODEL_WEIGHTS = {
    LSTM: 0.5,
    GRU: 0.3,
    XGBoost: 0.2
};

// State management
let chartInstance = null;
let currentPrediction = null;

// DOM Elements
const predictionForm = document.getElementById('predictionForm');
const cryptocurrencySelect = document.getElementById('cryptocurrencySelect');
const timeframeSelect = document.getElementById('timeframeSelect');
const klineCountInput = document.getElementById('klineCountInput');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const settingsBtn = document.getElementById('settingsBtn');
const loadingSpinner = document.getElementById('loadingSpinner');
const alertContainer = document.getElementById('alertContainer');
const resultsContainer = document.getElementById('resultsContainer');
const chartContainer = document.getElementById('chartContainer');

// Event Listeners
predictionForm.addEventListener('submit', handlePrediction);
cryptocurrencySelect.addEventListener('change', updateTimeframeOptions);
settingsBtn.addEventListener('click', showSettings);

// Update available timeframes based on selected cryptocurrency
function updateTimeframeOptions() {
    const selectedCrypto = cryptocurrencySelect.value;
    const allOptions = Array.from(timeframeSelect.options);
    const currentValue = timeframeSelect.value;

    if (selectedCrypto && CRYPTO_CONFIG[selectedCrypto]) {
        const supportedTimeframes = CRYPTO_CONFIG[selectedCrypto].timeframes;
        const isFullSupport = CRYPTO_CONFIG[selectedCrypto].fullSupport;

        allOptions.forEach(option => {
            if (option.value === '') {
                option.disabled = false;
            } else {
                option.disabled = !supportedTimeframes.includes(option.value);
                
                // Add label for limited support
                if (!isFullSupport && supportedTimeframes.includes(option.value)) {
                    option.textContent = option.textContent.replace(' - ', ' (Limited) - ');
                }
            }
        });

        // Reset timeframe selection if current is not supported
        if (currentValue && !supportedTimeframes.includes(currentValue)) {
            timeframeSelect.value = '';
        }
    }
}

// Handle prediction request
async function handlePrediction(e) {
    e.preventDefault();

    const crypto = cryptocurrencySelect.value;
    const timeframe = timeframeSelect.value;
    const klinesCount = parseInt(klineCountInput.value);

    // Validation
    if (!crypto || !timeframe || !klinesCount) {
        showAlert('Please fill in all required fields', 'error');
        return;
    }

    if (klinesCount < 20 || klinesCount > 1000) {
        showAlert('K-line count must be between 20 and 1000', 'error');
        return;
    }

    // Validate timeframe support
    if (!CRYPTO_CONFIG[crypto].timeframes.includes(timeframe)) {
        showAlert(`${timeframe} timeframe is not supported for ${crypto}`, 'error');
        return;
    }

    // Show loading state
    loadingSpinner.style.display = 'block';
    predictBtn.disabled = true;
    resultsContainer.innerHTML = '';
    chartContainer.style.display = 'none';

    try {
        const response = await fetch(`${API_BASE_URL}${API_ENDPOINTS.predict}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                symbol: crypto + 'USDT',
                timeframe: timeframe,
                klines: klinesCount
            })
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.statusText}`);
        }

        const data = await response.json();
        currentPrediction = data;
        displayResults(data);
        showAlert('Prediction completed successfully', 'success');
    } catch (error) {
        console.error('Prediction error:', error);
        showAlert(`Error: ${error.message}`, 'error');
    } finally {
        loadingSpinner.style.display = 'none';
        predictBtn.disabled = false;
    }
}

// Display prediction results
function displayResults(data) {
    resultsContainer.innerHTML = '';

    // Current Price Card
    const currentPriceCard = createCard(
        'Current Market Data',
        [
            { label: 'Symbol', value: data.symbol },
            { label: 'Current Price', value: `$${data.current_price.toFixed(8)}` },
            { label: 'Timeframe', value: data.timeframe },
            { label: 'K-lines Analyzed', value: data.klines_count }
        ]
    );
    resultsContainer.appendChild(currentPriceCard);

    // Prediction Card
    const predictionCard = createCard(
        'Price Prediction',
        [
            { 
                label: 'Predicted Price', 
                value: `$${data.predicted_price.toFixed(8)}`,
                highlight: data.predicted_price > data.current_price ? 'price-up' : 'price-down'
            },
            { 
                label: 'Price Change', 
                value: (() => {
                    const change = ((data.predicted_price - data.current_price) / data.current_price * 100);
                    return `${change > 0 ? '+' : ''}${change.toFixed(2)}%`;
                })(),
                highlight: data.predicted_price > data.current_price ? 'price-up' : 'price-down'
            },
            { label: 'Expected Accuracy', value: `${(ACCURACY_LEVELS[data.timeframe] * 100).toFixed(1)}%` }
        ]
    );
    resultsContainer.appendChild(predictionCard);

    // Risk Management Card
    const entryPrice = data.entry_price;
    const stopLoss = data.stop_loss;
    const takeProfit = data.take_profit;
    const riskReward = ((takeProfit - entryPrice) / (entryPrice - stopLoss)).toFixed(2);

    const riskCard = createCard(
        'Risk Management',
        [
            { label: 'Entry Price', value: `$${entryPrice.toFixed(8)}` },
            { label: 'Stop Loss', value: `$${stopLoss.toFixed(8)}` },
            { label: 'Take Profit', value: `$${takeProfit.toFixed(8)}` },
            { label: 'Risk/Reward Ratio', value: `1:${riskReward}` }
        ]
    );
    resultsContainer.appendChild(riskCard);

    // Signal & Confidence Card
    const confidence = data.confidence;
    const recommendation = data.recommendation;
    const signalCard = document.createElement('div');
    signalCard.className = 'card';
    signalCard.innerHTML = `
        <div class="card-title">Trading Signal & Confidence</div>
        <div class="metric">
            <span class="metric-label">Recommendation</span>
            <span class="metric-value">
                <span class="badge badge-${recommendation.toLowerCase()}">${recommendation}</span>
            </span>
        </div>
        <div class="metric">
            <span class="metric-label">Confidence Level</span>
            <span class="metric-value">${(confidence * 100).toFixed(1)}%</span>
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: ${confidence * 100}%"></div>
        </div>
        <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(138, 43, 226, 0.1);">
            <div class="metric">
                <span class="metric-label">RSI</span>
                <span class="metric-value">${data.technical_indicators.RSI.toFixed(2)}</span>
            </div>
            <div class="metric">
                <span class="metric-label">MACD</span>
                <span class="metric-value" style="color: ${data.technical_indicators.MACD > 0 ? '#00ff41' : '#ff3b30'};">
                    ${data.technical_indicators.MACD > 0 ? '↑' : '↓'} ${Math.abs(data.technical_indicators.MACD).toFixed(6)}
                </span>
            </div>
            <div class="metric">
                <span class="metric-label">ADX</span>
                <span class="metric-value">${data.technical_indicators.ADX.toFixed(2)}</span>
            </div>
        </div>
    `;
    resultsContainer.appendChild(signalCard);

    // Volatility Card
    const currentVol = data.volatility.current;
    const predictedVol = data.volatility.predicted;
    const volLevel = predictedVol < 0.005 ? 'Low' : (predictedVol < 0.015 ? 'Medium' : 'High');

    const volatilityCard = createCard(
        'Volatility Assessment',
        [
            { label: 'Current Volatility', value: `${(currentVol * 100).toFixed(3)}%` },
            { label: 'Predicted Volatility', value: `${(predictedVol * 100).toFixed(3)}%` },
            { label: 'Volatility Level', value: volLevel },
            { label: 'Change', value: `${((predictedVol - currentVol) * 100).toFixed(3)}%` }
        ]
    );
    resultsContainer.appendChild(volatilityCard);

    // Model Distribution Card
    const modelCard = document.createElement('div');
    modelCard.className = 'card';
    modelCard.innerHTML = `
        <div class="card-title">Model Predictions</div>
        <div class="metric">
            <span class="metric-label">LSTM (50%)</span>
            <span class="metric-value">$${data.model_predictions.LSTM.toFixed(8)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">GRU (30%)</span>
            <span class="metric-value">$${data.model_predictions.GRU.toFixed(8)}</span>
        </div>
        <div class="metric">
            <span class="metric-label">XGBoost (20%)</span>
            <span class="metric-value">$${data.model_predictions.XGBoost.toFixed(8)}</span>
        </div>
        <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(138, 43, 226, 0.1);">
            <div class="metric">
                <span class="metric-label">Final Prediction</span>
                <span class="metric-value">$${data.predicted_price.toFixed(8)}</span>
            </div>
        </div>
    `;
    resultsContainer.appendChild(modelCard);

    // Display chart
    displayChart(data);
}

// Create a metric card
function createCard(title, metrics) {
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `<div class="card-title">${title}</div>`;

    metrics.forEach(metric => {
        const metricDiv = document.createElement('div');
        metricDiv.className = 'metric';
        metricDiv.innerHTML = `
            <span class="metric-label">${metric.label}</span>
            <span class="metric-value ${metric.highlight || ''}">${metric.value}</span>
        `;
        card.appendChild(metricDiv);
    });

    return card;
}

// Display price prediction chart
function displayChart(data) {
    chartContainer.style.display = 'block';
    const ctx = document.getElementById('predictionChart').getContext('2d');

    // Historical prices (simulated)
    const historicalPrices = data.historical_prices || generateHistoricalPrices(
        data.current_price,
        data.klines_count
    );

    // Create chart data
    const chartData = {
        labels: historicalPrices.map((_, i) => `K${i + 1}`),
        datasets: [
            {
                label: 'Historical Price',
                data: historicalPrices,
                borderColor: '#8a2be2',
                backgroundColor: 'rgba(138, 43, 226, 0.1)',
                borderWidth: 2,
                tension: 0.4,
                fill: true,
                pointRadius: 0,
                pointHoverRadius: 5
            },
            {
                label: 'Predicted Price',
                data: [...historicalPrices.slice(-5), data.predicted_price],
                borderColor: '#00bfff',
                backgroundColor: 'rgba(0, 191, 255, 0.1)',
                borderWidth: 2,
                borderDash: [5, 5],
                fill: false,
                pointRadius: 4,
                pointBackgroundColor: '#00bfff'
            }
        ]
    };

    // Destroy existing chart if it exists
    if (chartInstance) {
        chartInstance.destroy();
    }

    // Create new chart
    chartInstance = new Chart(ctx, {
        type: 'line',
        data: chartData,
        options: {
            responsive: true,
            maintainAspectRatio: true,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#fff',
                        font: { size: 12 }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: '#8a2be2',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(138, 43, 226, 0.1)'
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)',
                        maxTicksLimit: 10
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(138, 43, 226, 0.1)'
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)',
                        callback: function(value) {
                            return '$' + value.toFixed(2);
                        }
                    }
                }
            }
        }
    });
}

// Generate simulated historical prices for visualization
function generateHistoricalPrices(currentPrice, count) {
    const prices = [currentPrice];
    let price = currentPrice;

    for (let i = 1; i < count; i++) {
        const change = (Math.random() - 0.5) * 0.02;
        price = price * (1 + change);
        prices.unshift(price);
    }

    return prices;
}

// Show alert message
function showAlert(message, type = 'info') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} show`;
    alertDiv.textContent = message;
    alertContainer.innerHTML = '';
    alertContainer.appendChild(alertDiv);

    setTimeout(() => {
        alertDiv.style.opacity = '0';
        setTimeout(() => alertDiv.remove(), 300);
    }, 5000);
}

// Settings modal (placeholder)
function showSettings() {
    alert('Settings Panel - V6 Model Configuration\n\n' +
        'LSTM Weight: 50%\n' +
        'GRU Weight: 30%\n' +
        'XGBoost Weight: 20%\n\n' +
        'API: http://localhost:8001\n\n' +
        'Advanced settings coming soon...');
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('CPB Crypto Predictor V6 loaded');
    console.log('API Base URL:', API_BASE_URL);
    console.log('Supported cryptocurrencies:', Object.keys(CRYPTO_CONFIG).length);
});
