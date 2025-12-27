#!/usr/bin/env python3
"""
Price Prediction Visualization Module
Provides improved chart generation for prediction results

Features:
- Fixed data consistency: Always shows EXACTLY the requested klines
- JavaScript displays exactly what Python sends (no filtering that causes shifting)
- No smoothing (tension: 0)
- Better color schemes
- Responsive design
"""

import json
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class ChartGenerator:
    """Generate interactive charts for price predictions"""
    
    @staticmethod
    def generate_price_chart(prices, predicted_price, symbol, timeframe):
        """
        Generate interactive price prediction chart using Chart.js
        
        CRITICAL FIX: This function receives EXACTLY the klines_count prices from app.py
        and displays ALL of them, no filtering or shifting
        
        Args:
            prices: List of historical prices (EXACTLY klines_count items)
            predicted_price: Predicted next price
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Time frame (e.g., '1d')
        
        Returns:
            HTML string with embedded chart (fragment, no body tags)
        """
        
        # Input validation and logging
        if not prices or len(prices) == 0:
            logger.error("No prices provided to chart generation")
            raise ValueError("Prices list is empty")
        
        prices_list = list(prices)
        n_total = len(prices_list)
        logger.info(f"Chart generation: symbol={symbol}, timeframe={timeframe}, prices_count={n_total}")
        
        # Calculate moving averages on FULL dataset (all prices)
        ma7_full = ChartGenerator._calculate_ma(prices_list, 7)
        ma21_full = ChartGenerator._calculate_ma(prices_list, 21)
        
        # Current price (last price point)
        current_price = prices_list[-1]
        
        # Log data consistency
        logger.info(f"Data consistency check:")
        logger.info(f"  - Total prices received: {n_total}")
        logger.info(f"  - MA7 points: {len([x for x in ma7_full if x is not None])}")
        logger.info(f"  - MA21 points: {len([x for x in ma21_full if x is not None])}")
        logger.info(f"  - Current price: ${current_price:.2f}")
        logger.info(f"  - Predicted price: ${predicted_price:.2f}")
        
        # Prepare data for JSON
        prices_json = json.dumps([float(p) for p in prices_list])
        ma7_json = json.dumps([None if x is None else float(x) for x in ma7_full])
        ma21_json = json.dumps([None if x is None else float(x) for x in ma21_full])
        predicted_json = json.dumps(float(predicted_price))
        current_json = json.dumps(float(current_price))
        
        # Build HTML with simple string formatting
        html = f"""
<style>
    .chart-wrapper {{
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 30px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }}
    
    .chart-legend {{
        display: flex;
        gap: 30px;
        justify-content: center;
        margin-bottom: 20px;
        flex-wrap: wrap;
    }}
    
    .legend-item {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 14px;
    }}
    
    .legend-color {{
        width: 16px;
        height: 16px;
        border-radius: 2px;
    }}
    
    .legend-historical {{
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
    }}
    
    .legend-predicted {{
        background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
        border: 2px dashed currentColor;
        opacity: 0.7;
    }}
    
    .legend-ma7 {{
        background: rgba(34, 197, 94, 0.6);
    }}
    
    .legend-ma21 {{
        background: rgba(245, 158, 11, 0.6);
    }}
    
    .chart-container {{
        position: relative;
        width: 100%;
        height: 400px;
        margin-bottom: 15px;
    }}
    
    .data-info {{
        font-size: 12px;
        color: #64748b;
        text-align: center;
        margin-top: 15px;
        padding: 10px;
        background: rgba(100, 150, 200, 0.1);
        border-radius: 8px;
    }}
</style>

<div class="chart-wrapper">
    <div class="chart-legend">
        <div class="legend-item">
            <div class="legend-color legend-historical"></div>
            <span>Historical Price</span>
        </div>
        <div class="legend-item">
            <div class="legend-color legend-predicted"></div>
            <span>Predicted Price</span>
        </div>
        <div class="legend-item">
            <div class="legend-color legend-ma7"></div>
            <span>MA(7)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color legend-ma21"></div>
            <span>MA(21)</span>
        </div>
    </div>
    
    <div class="chart-container">
        <canvas id="priceChart"></canvas>
    </div>
    
    <div class="data-info">
        <strong>Data Consistency Info:</strong> Displaying exactly {n_total} historical candles + 1 predicted point
    </div>
</div>

<script>
(function() {{
    // Full data from server - EXACTLY what the API sent
    const fullPrices = {prices_json};
    const ma7Data = {ma7_json};
    const ma21Data = {ma21_json};
    const predictedPrice = {predicted_json};
    const currentPrice = {current_json};
    
    // CRITICAL FIX: Display ALL prices received, no filtering
    const displayPrices = fullPrices;
    const displayMA7 = ma7Data;
    const displayMA21 = ma21Data;
    
    // Generate labels for ALL data points
    const labels = [];
    for (let i = 0; i < displayPrices.length; i++) {{
        labels.push('K' + (i + 1));
    }}
    labels.push('K+1');
    
    console.log('[CHART] Data consistency check (JavaScript):');
    console.log('[CHART]   - Full prices length: ' + fullPrices.length);
    console.log('[CHART]   - Display prices length: ' + displayPrices.length);
    console.log('[CHART]   - Labels length: ' + labels.length);
    
    // Prepare historical prices
    const historicalPrices = displayPrices;
    
    // Prepare predicted prices
    const predictedPrices = [];
    for (let i = 0; i < displayPrices.length; i++) {{
        predictedPrices.push(null);
    }}
    predictedPrices.push(currentPrice);
    predictedPrices.push(predictedPrice);
    
    // Prepare MA data
    const ma7Display = displayMA7.concat([null]);
    const ma21Display = displayMA21.concat([null]);
    
    // Initialize chart
    const initChart = function() {{
        if (typeof Chart === 'undefined') {{
            console.log('[CHART] Waiting for Chart.js to load...');
            setTimeout(initChart, 100);
            return;
        }}
        
        try {{
            const canvas = document.getElementById('priceChart');
            if (!canvas) {{
                console.error('[CHART] Canvas element not found');
                return;
            }}
            
            const ctx = canvas.getContext('2d');
            if (!ctx) {{
                console.error('[CHART] Failed to get canvas context');
                return;
            }}
            
            console.log('[CHART] Creating Chart.js instance...');
            
            const chart = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: labels,
                    datasets: [
                        {{
                            label: 'Historical Price',
                            data: historicalPrices,
                            borderColor: '#a855f7',
                            backgroundColor: 'rgba(168, 85, 247, 0.1)',
                            borderWidth: 3,
                            pointRadius: 4,
                            pointBackgroundColor: '#7c3aed',
                            pointBorderColor: '#fff',
                            pointBorderWidth: 2,
                            pointHoverRadius: 6,
                            tension: 0,
                            fill: true,
                            spanGaps: false
                        }},
                        {{
                            label: 'Predicted Price',
                            data: predictedPrices,
                            borderColor: '#06b6d4',
                            backgroundColor: 'rgba(6, 182, 212, 0.05)',
                            borderWidth: 3,
                            borderDash: [5, 5],
                            pointRadius: 6,
                            pointBackgroundColor: '#0891b2',
                            pointBorderColor: '#fff',
                            pointBorderWidth: 2,
                            pointHoverRadius: 8,
                            tension: 0,
                            fill: false,
                            spanGaps: false
                        }},
                        {{
                            label: 'MA(7)',
                            data: ma7Display,
                            borderColor: 'rgba(34, 197, 94, 0.6)',
                            backgroundColor: 'transparent',
                            borderWidth: 2,
                            pointRadius: 0,
                            tension: 0,
                            fill: false,
                            spanGaps: true
                        }},
                        {{
                            label: 'MA(21)',
                            data: ma21Display,
                            borderColor: 'rgba(245, 158, 11, 0.6)',
                            backgroundColor: 'transparent',
                            borderWidth: 2,
                            pointRadius: 0,
                            tension: 0,
                            fill: false,
                            spanGaps: true
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {{
                        mode: 'index',
                        intersect: false
                    }},
                    plugins: {{
                        legend: {{
                            display: false
                        }},
                        tooltip: {{
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            titleColor: '#fff',
                            bodyColor: '#e0e0e0',
                            borderColor: 'rgba(255, 255, 255, 0.2)',
                            borderWidth: 1,
                            padding: 12
                        }}
                    }},
                    scales: {{
                        x: {{
                            grid: {{
                                color: 'rgba(255, 255, 255, 0.05)',
                                drawBorder: false
                            }},
                            ticks: {{
                                color: 'rgba(255, 255, 255, 0.6)'
                            }}
                        }},
                        y: {{
                            grid: {{
                                color: 'rgba(255, 255, 255, 0.05)',
                                drawBorder: false
                            }},
                            ticks: {{
                                color: 'rgba(255, 255, 255, 0.6)'
                            }}
                        }}
                    }}
                }}
            }});
            
            console.log('[CHART] Chart rendered successfully');
        }} catch (err) {{
            console.error('[CHART] Error creating chart:', err);
        }}
    }};
    
    // Initialize chart when DOM is ready
    if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', initChart);
    }} else {{
        initChart();
    }}
}})();
</script>
"""
        return html
    
    @staticmethod
    def _calculate_ma(prices, period):
        """Calculate moving average"""
        prices = np.array(prices, dtype=float)
        ma = np.full(len(prices), np.nan)
        
        if len(prices) >= period:
            for i in range(period - 1, len(prices)):
                ma[i] = np.mean(prices[i - period + 1:i + 1])
        
        return ma.tolist()
    
    @staticmethod
    def generate_technical_chart(technical_indicators):
        """
        Generate technical indicators dashboard
        
        Args:
            technical_indicators: Dict with RSI, MACD, ATR, etc.
        
        Returns:
            HTML string with indicators dashboard (fragment, no body tags)
        """
        
        rsi = technical_indicators.get('RSI', 0)
        macd = technical_indicators.get('MACD', 0)
        adx = technical_indicators.get('ADX', 0)
        atr = technical_indicators.get('ATR', 0)
        volatility = technical_indicators.get('Volatility', 0)
        
        rsi_desc = ChartGenerator._get_rsi_description(rsi)
        macd_desc = ChartGenerator._get_macd_description(macd)
        adx_desc = ChartGenerator._get_adx_description(adx)
        
        html = f"""
<style>
    .indicators-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 20px;
    }}
    
    .indicator-card {{
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 25px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }}
    
    .indicator-title {{
        font-size: 14px;
        font-weight: 600;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 15px;
    }}
    
    .indicator-value {{
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 15px;
        background: linear-gradient(135deg, #7c3aed 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    
    .indicator-bar {{
        height: 8px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        overflow: hidden;
        margin-bottom: 8px;
    }}
    
    .indicator-fill {{
        height: 100%;
        background: linear-gradient(90deg, #7c3aed 0%, #06b6d4 100%);
        transition: width 0.3s ease;
    }}
    
    .indicator-description {{
        font-size: 12px;
        color: #64748b;
        line-height: 1.6;
    }}
</style>

<h2 style="margin-bottom: 30px;">Technical Indicators Analysis</h2>

<div class="indicators-grid">
    <div class="indicator-card">
        <div class="indicator-title">RSI (14)</div>
        <div class="indicator-value">{rsi:.1f}</div>
        <div class="indicator-bar">
            <div class="indicator-fill" style="width: {min(100, rsi)}%"></div>
        </div>
        <div class="indicator-description">{rsi_desc}</div>
    </div>
    
    <div class="indicator-card">
        <div class="indicator-title">MACD</div>
        <div class="indicator-value">{macd:.4f}</div>
        <div class="indicator-bar">
            <div class="indicator-fill" style="width: {min(100, max(0, macd * 50 + 50))}%"></div>
        </div>
        <div class="indicator-description">{macd_desc}</div>
    </div>
    
    <div class="indicator-card">
        <div class="indicator-title">ADX (14)</div>
        <div class="indicator-value">{adx:.1f}</div>
        <div class="indicator-bar">
            <div class="indicator-fill" style="width: {min(100, adx)}%"></div>
        </div>
        <div class="indicator-description">{adx_desc}</div>
    </div>
    
    <div class="indicator-card">
        <div class="indicator-title">ATR (14)</div>
        <div class="indicator-value">{atr:.4f}</div>
        <div class="indicator-description">Average True Range for volatility assessment</div>
    </div>
    
    <div class="indicator-card">
        <div class="indicator-title">Volatility</div>
        <div class="indicator-value">{volatility:.4f}</div>
        <div class="indicator-bar">
            <div class="indicator-fill" style="width: {min(100, volatility * 1000)}%"></div>
        </div>
        <div class="indicator-description">Historical price volatility</div>
    </div>
</div>
        """
        return html
    
    @staticmethod
    def _get_rsi_description(rsi):
        if rsi >= 70:
            return "Overbought Signal - Potential Sell Pressure"
        elif rsi <= 30:
            return "Oversold Signal - Potential Buy Opportunity"
        elif rsi >= 50:
            return "Bullish Momentum Building"
        else:
            return "Bearish Momentum Building"
    
    @staticmethod
    def _get_macd_description(macd):
        if macd > 0:
            return "Positive MACD - Bullish Signal"
        elif macd < 0:
            return "Negative MACD - Bearish Signal"
        else:
            return "MACD at Zero - Neutral"
    
    @staticmethod
    def _get_adx_description(adx):
        if adx >= 50:
            return "Very Strong Trend"
        elif adx >= 30:
            return "Strong Trend"
        elif adx >= 20:
            return "Weak Trend"
        else:
            return "No Trend - Range Bound"
