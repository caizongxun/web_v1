#!/usr/bin/env python3
"""
Price Prediction Visualization Module
Provides improved chart generation for prediction results

Features:
- Display last 40 candles from requested data
- Leave space for prediction line
- No smoothing (tension: 0)
- Better color schemes
- Proper legend positioning
- Technical indicators overlay
- Responsive design
"""

import json
import numpy as np
from datetime import datetime, timedelta

class ChartGenerator:
    """Generate interactive charts for price predictions"""
    
    @staticmethod
    def generate_price_chart(prices, predicted_price, symbol, timeframe):
        """
        Generate interactive price prediction chart using Chart.js
        
        Args:
            prices: List of historical prices (e.g., 100 points)
            predicted_price: Predicted next price
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Time frame (e.g., '1d')
        
        Returns:
            HTML string with embedded chart
        """
        
        # Only display last 40 candles from the requested data
        # This leaves room for the prediction line
        display_count = 40
        start_idx = max(0, len(prices) - display_count)
        
        # Get the portion to display
        displayed_prices = prices[start_idx:]
        n_display = len(displayed_prices)
        
        # Generate K-line labels (show only the displayed portion)
        # K1 represents the oldest displayed candle
        labels = [f'K{start_idx + i + 1}' for i in range(n_display)] + ['K+1']
        
        # Current price (last price point)
        current_price = prices[-1]
        
        # Prepare data: only show displayed historical prices, then predicted
        historical_prices = list(displayed_prices)
        predicted_prices = [None] * n_display + [current_price, predicted_price]
        
        # Calculate moving averages on FULL dataset
        ma7_full = ChartGenerator._calculate_ma(prices, 7)
        ma21_full = ChartGenerator._calculate_ma(prices, 21)
        
        # Extract only the displayed portion of MAs
        ma7_display = ma7_full[start_idx:] + [None]
        ma21_display = ma21_full[start_idx:] + [None]
        
        # HTML with Chart.js
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Price Prediction - {symbol}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            background: linear-gradient(135deg, #1a1b26 0%, #2d2e42 100%);
            color: #e0e0e0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            font-size: 32px;
            font-weight: 600;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #7c3aed 0%, #06b6d4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .header p {{
            color: #a0aec0;
            font-size: 14px;
        }}
        
        .chart-wrapper {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 30px;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }}
        
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }}
        
        .info-card {{
            background: rgba(124, 58, 237, 0.1);
            border: 1px solid rgba(124, 58, 237, 0.3);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }}
        
        .info-card .label {{
            color: #a0aec0;
            font-size: 12px;
            text-transform: uppercase;
            margin-bottom: 8px;
            letter-spacing: 1px;
        }}
        
        .info-card .value {{
            font-size: 24px;
            font-weight: 600;
            color: #06b6d4;
        }}
        
        .legend {{
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
        
        canvas {{
            max-height: 400px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Price Prediction Visualization</h1>
            <p>Advanced Technical Analysis for {symbol} ({timeframe}) - Showing Last {n_display} Candles</p>
        </div>
        
        <div class="chart-wrapper">
            <div class="legend">
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
            
            <canvas id="priceChart"></canvas>
        </div>
        
        <div class="info-grid">
            <div class="info-card">
                <div class="label">Current Price</div>
                <div class="value">${{{current_price}:,.2f}}</div>
            </div>
            <div class="info-card">
                <div class="label">Predicted Price</div>
                <div class="value">${{{predicted_price}:,.2f}}</div>
            </div>
            <div class="info-card">
                <div class="label">Change Direction</div>
                <div class="value" style="color: {('#06b6d4' if predicted_price > current_price else '#ef4444')}">
                    {('↑ UP' if predicted_price > current_price else '↓ DOWN')}
                </div>
            </div>
            <div class="info-card">
                <div class="label">Change Percent</div>
                <div class="value" style="color: {('#06b6d4' if predicted_price > current_price else '#ef4444')}">
                    {abs((predicted_price - current_price) / current_price * 100):.2f}%
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const ctx = document.getElementById('priceChart').getContext('2d');
        const labels = {json.dumps(labels)};
        const historicalPrices = {json.dumps(historical_prices)};
        const predictedPrices = {json.dumps(predicted_prices)};
        const ma7Data = {json.dumps([None if x is None else float(x) for x in ma7_display])};
        const ma21Data = {json.dumps([None if x is None else float(x) for x in ma21_display])};
        
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
                        data: ma7Data,
                        borderColor: 'rgba(34, 197, 94, 0.6)',
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        tension: 0,
                        fill: false,
                        spanGaps: true
                    }},
                    {{
                        label: 'MA(21)',
                        data: ma21Data,
                        borderColor: 'rgba(245, 158, 11, 0.6)',
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        tension: 0,
                        fill: false,
                        spanGaps: true
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: true,
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
                        padding: 12,
                        titleFont: {{
                            size: 14,
                            weight: 'bold'
                        }},
                        bodyFont: {{
                            size: 12
                        }},
                        callbacks: {{
                            label: function(context) {{
                                let label = context.dataset.label || '';
                                if (label) {{
                                    label += ': ';
                                }}
                                if (context.parsed.y !== null) {{
                                    label += '$' + context.parsed.y.toLocaleString(undefined, {{
                                        minimumFractionDigits: 2,
                                        maximumFractionDigits: 2
                                    }});
                                }}
                                return label;
                            }}
                        }}
                    }}
                }},
                scales: {{
                    x: {{
                        grid: {{
                            color: 'rgba(255, 255, 255, 0.05)',
                            drawBorder: false
                        }},
                        ticks: {{
                            color: 'rgba(255, 255, 255, 0.6)',
                            font: {{
                                size: 12
                            }}
                        }}
                    }},
                    y: {{
                        grid: {{
                            color: 'rgba(255, 255, 255, 0.05)',
                            drawBorder: false
                        }},
                        ticks: {{
                            color: 'rgba(255, 255, 255, 0.6)',
                            font: {{
                                size: 12
                            }},
                            callback: function(value) {{
                                return '$' + value.toLocaleString();
                            }}
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
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
            HTML string with indicators dashboard
        """
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Technical Indicators</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            background: linear-gradient(135deg, #1a1b26 0%, #2d2e42 100%);
            color: #e0e0e0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            font-size: 32px;
            font-weight: 600;
            margin-bottom: 10px;
            background: linear-gradient(135deg, #7c3aed 0%, #06b6d4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
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
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Technical Indicators Analysis</h1>
        </div>
        
        <div class="indicators-grid">
            <!-- RSI -->
            <div class="indicator-card">
                <div class="indicator-title">RSI (14)</div>
                <div class="indicator-value">{technical_indicators.get('RSI', 0):.1f}</div>
                <div class="indicator-bar">
                    <div class="indicator-fill" style="width: {min(100, technical_indicators.get('RSI', 0))}%"></div>
                </div>
                <div class="indicator-description">
                    {ChartGenerator._get_rsi_description(technical_indicators.get('RSI', 0))}
                </div>
            </div>
            
            <!-- MACD -->
            <div class="indicator-card">
                <div class="indicator-title">MACD</div>
                <div class="indicator-value">{technical_indicators.get('MACD', 0):.4f}</div>
                <div class="indicator-bar">
                    <div class="indicator-fill" style="width: {min(100, max(0, technical_indicators.get('MACD', 0) * 50 + 50))}%"></div>
                </div>
                <div class="indicator-description">
                    {ChartGenerator._get_macd_description(technical_indicators.get('MACD', 0))}
                </div>
            </div>
            
            <!-- ADX -->
            <div class="indicator-card">
                <div class="indicator-title">ADX (14)</div>
                <div class="indicator-value">{technical_indicators.get('ADX', 0):.1f}</div>
                <div class="indicator-bar">
                    <div class="indicator-fill" style="width: {min(100, technical_indicators.get('ADX', 0))}%"></div>
                </div>
                <div class="indicator-description">
                    {ChartGenerator._get_adx_description(technical_indicators.get('ADX', 0))}
                </div>
            </div>
            
            <!-- ATR -->
            <div class="indicator-card">
                <div class="indicator-title">ATR (14)</div>
                <div class="indicator-value">{technical_indicators.get('ATR', 0):.4f}</div>
                <div class="indicator-description">Average True Range for volatility assessment</div>
            </div>
            
            <!-- Volatility -->
            <div class="indicator-card">
                <div class="indicator-title">Volatility</div>
                <div class="indicator-value">{technical_indicators.get('Volatility', 0):.4f}</div>
                <div class="indicator-bar">
                    <div class="indicator-fill" style="width: {min(100, technical_indicators.get('Volatility', 0) * 1000)}%"></div>
                </div>
                <div class="indicator-description">Historical price volatility</div>
            </div>
        </div>
    </div>
</body>
</html>
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


if __name__ == "__main__":
    # Example usage
    sample_prices = [82000, 82500, 82200, 82800, 82300, 83000, 82900, 83200, 83100]
    sample_predicted = 83500
    
    html = ChartGenerator.generate_price_chart(sample_prices, sample_predicted, "BTCUSDT", "1d")
    
    with open("price_chart.html", "w") as f:
        f.write(html)
    
    print("Chart generated: price_chart.html")
