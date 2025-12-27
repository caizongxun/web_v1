// API Configuration
const API_BASE_URL = 'http://localhost:8001';
const API_ENDPOINTS = {
    predict: '/api/v6/predict',
    chart: '/api/v6/chart',
    technical: '/api/v6/indicators',
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

    console.log(`[PREDICTION] Starting prediction: ${crypto} ${timeframe} (${klinesCount}K)`);

    try {
        // Step 1: Get prediction data
        console.log(`[API] Calling ${API_BASE_URL}${API_ENDPOINTS.predict}`);
        const predictResponse = await fetch(`${API_BASE_URL}${API_ENDPOINTS.predict}`, {
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

        if (!predictResponse.ok) {
            throw new Error(`Predict API Error: ${predictResponse.statusText}`);
        }

        const predictData = await predictResponse.json();
        currentPrediction = predictData;
        console.log(`[PREDICTION] Received data:`, {
            symbol: predictData.symbol,
            timeframe: predictData.timeframe,
            klines_count: predictData.klines_count,
            predicted_price: predictData.predicted_price
        });
        
        // Display text results
        displayResults(predictData);
        showAlert('Prediction completed successfully', 'success');

        // Step 2: Get chart visualization
        console.log(`[CHART] Calling ${API_BASE_URL}${API_ENDPOINTS.chart}`);
        const chartResponse = await fetch(`${API_BASE_URL}${API_ENDPOINTS.chart}`, {
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

        if (!chartResponse.ok) {
            throw new Error(`Chart API Error: ${chartResponse.statusText}`);
        }

        const chartHtml = await chartResponse.text();
        console.log(`[CHART] Received HTML chart (${chartHtml.length} bytes)`);
        displayChartHTML(chartHtml);

        // Step 3: Get technical indicators
        console.log(`[INDICATORS] Calling ${API_BASE_URL}${API_ENDPOINTS.technical}`);
        const indicatorsResponse = await fetch(`${API_BASE_URL}${API_ENDPOINTS.technical}`, {
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

        if (!indicatorsResponse.ok) {
            console.warn(`Indicators API warning: ${indicatorsResponse.statusText}`);
        } else {
            const indicatorsHtml = await indicatorsResponse.text();
            console.log(`[INDICATORS] Received HTML indicators (${indicatorsHtml.length} bytes)`);
            displayIndicatorsHTML(indicatorsHtml);
        }

    } catch (error) {
        console.error('[ERROR]', error);
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
}

// Display chart as HTML (from /api/v6/chart) - FIXED HTML insertion
function displayChartHTML(html) {
    console.log('[CHART] Rendering chart HTML');
    chartContainer.style.display = 'block';
    
    try {
        // Use innerHTML directly - simpler and avoids script parsing issues
        chartContainer.innerHTML = html;
        console.log('[CHART] Chart HTML inserted successfully');
        
        // Ensure scripts in the container are executed
        const scripts = chartContainer.querySelectorAll('script');
        console.log(`[CHART] Found ${scripts.length} script tags`);
        
        scripts.forEach((script) => {
            try {
                // Create a new script element and copy attributes
                const newScript = document.createElement('script');
                
                // Copy all attributes
                Array.from(script.attributes).forEach(attr => {
                    newScript.setAttribute(attr.name, attr.value);
                });
                
                // Copy content
                if (script.textContent) {
                    newScript.textContent = script.textContent;
                }
                
                // Insert and execute
                document.body.appendChild(newScript);
                console.log('[CHART] Script executed');
            } catch (err) {
                console.error('[CHART] Script execution error:', err);
            }
        });
        
    } catch (err) {
        console.error('[CHART] Failed to insert chart HTML:', err);
        chartContainer.innerHTML = `<div style="color: red; padding: 20px;">Error rendering chart: ${err.message}</div>`;
    }
}

// Display indicators as HTML (from /api/v6/indicators)
function displayIndicatorsHTML(html) {
    console.log('[INDICATORS] Rendering indicators HTML');
    try {
        const indicatorsContainer = document.createElement('div');
        indicatorsContainer.style.marginTop = '30px';
        indicatorsContainer.innerHTML = html;
        resultsContainer.appendChild(indicatorsContainer);
        console.log('[INDICATORS] Indicators inserted successfully');
        
        // Execute any scripts in indicators
        const scripts = indicatorsContainer.querySelectorAll('script');
        scripts.forEach((script) => {
            try {
                const newScript = document.createElement('script');
                Array.from(script.attributes).forEach(attr => {
                    newScript.setAttribute(attr.name, attr.value);
                });
                if (script.textContent) {
                    newScript.textContent = script.textContent;
                }
                document.body.appendChild(newScript);
            } catch (err) {
                console.error('[INDICATORS] Script execution error:', err);
            }
        });
    } catch (err) {
        console.error('[INDICATORS] Failed to insert indicators HTML:', err);
    }
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
