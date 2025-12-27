#!/usr/bin/env python3
"""
Price Prediction Visualization Module
Provides improved chart generation for prediction results

Features:
- Fixed data consistency: Always shows EXACTLY the requested klines
- JavaScript displays exactly what Python sends (no filtering that causes shifting)
- No smoothing (tension: 0)
- Better color schemes
- Proper legend positioning
- Technical indicators overlay
- Responsive design

CRITICAL: This version ensures 100% data consistency
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
            HTML string with embedded chart
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
        
        # HTML with Chart.js - NO filtering in JavaScript, display exactly what we have
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Price Prediction - {symbol}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"><\/script>
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
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Price Prediction Visualization</h1>
            <p>Advanced Technical Analysis for {symbol} ({timeframe}) - Total {n_total} Candles</p>
        </div>
        
        <div class="chart-wrapper">
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color legend-historical"><\/div>
                    <span>Historical Price<\/span>
                <\/div>
                <div class="legend-item">
                    <div class="legend-color legend-predicted"><\/div>
                    <span>Predicted Price<\/span>
                <\/div>
                <div class="legend-item">
                    <div class="legend-color legend-ma7"><\/div>
                    <span>MA(7)<\/span>
                <\/div>
                <div class="legend-item">
                    <div class="legend-color legend-ma21"><\/div>
                    <span>MA(21)<\/span>
                <\/div>
            <\/div>
            
            <canvas id="priceChart"><\/canvas>
            
            <div class="data-info">
                <strong>Data Consistency Info:<\/strong> Displaying exactly {n_total} historical candles + 1 predicted point
            <\/div>
        <\/div>
        
        <div class="info-grid">
            <div class="info-card">
                <div class="label">Current Price<\/div>
                <div class="value">${{{current_price}:,.2f}}<\/div>
            <\/div>
            <div class="info-card">
                <div class="label">Predicted Price<\/div>
                <div class="value">${{{predicted_price}:,.2f}}<\/div>
            <\/div>
            <div class="info-card">
                <div class="label">Change Direction<\/div>
                <div class="value" style="color: {('#06b6d4' if predicted_price > current_price else '#ef4444')}">
                    {('↑ UP' if predicted_price > current_price else '↓ DOWN')}
                <\/div>
            <\/div>
            <div class="info-card">
                <div class="label">Change Percent<\/div>
                <div class="value" style="color: {('#06b6d4' if predicted_price > current_price else '#ef4444')}">
                    {abs((predicted_price - current_price) / current_price * 100):.2f}%
                <\/div>
            <\/div>
        <\/div>
    <\/div>
    
    <script>
        // Full data from server - EXACTLY what the API sent
        const fullPrices = {json.dumps([float(p) for p in prices_list])};
        const ma7Data = {json.dumps([None if x is None else float(x) for x in ma7_full])};
        const ma21Data = {json.dumps([None if x is None else float(x) for x in ma21_full])};
        const predictedPrice = {json.dumps(float(predicted_price))};
        const currentPrice = {json.dumps(float(current_price))};
        
        // CRITICAL FIX: Display ALL prices received, no filtering
        const displayPrices = fullPrices;  // Use ALL prices
        const displayMA7 = ma7Data;        // Use ALL MA7 points
        const displayMA21 = ma21Data;      // Use ALL MA21 points
        
        // Generate labels for ALL data points
        const labels = [];
        for (let i = 0; i < displayPrices.length; i++) {{
            labels.push(`K${{i + 1}}`);
        }}
        labels.push('K+1'); // Prediction point
        
        console.log('Data consistency check (JavaScript):');
        console.log(`  - Full prices length: ${{fullPrices.length}}`);
        console.log(`  - Display prices length: ${{displayPrices.length}}`);
        console.log(`  - Labels length: ${{labels.length}}`);
        console.log(`  - Current price: ${{currentPrice.toFixed(2)}}`);
        console.log(`  - Predicted price: ${{predictedPrice.toFixed(2)}}`);
        
        // Prepare historical prices (all received data)
        const historicalPrices = displayPrices;
        
        // Prepare predicted prices (None for historical, then current + predicted)
        const predictedPrices = [];
        for (let i = 0; i < displayPrices.length; i++) {{
            predictedPrices.push(null);
        }}
        predictedPrices.push(currentPrice);
        predictedPrices.push(predictedPrice);
        
        // Prepare MA data (extend with one null for prediction point)
        const ma7Display = displayMA7.concat([null]);
        const ma21Display = displayMA21.concat([null]);
        
        // Chart.js configuration
        const ctx = document.getElementById('priceChart').getContext('2d');
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
                        tension: 0,  // NO SMOOTHING
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
                        tension: 0,  // NO SMOOTHING
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
                        pointHoverRadius: 4,
                        tension: 0,  // NO SMOOTHING
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
                        pointHoverRadius: 4,
                        tension: 0,  // NO SMOOTHING
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
        
        console.log('Chart rendered successfully with', displayPrices.length, 'historical candles');
    <\/script>
<\/body>
<\/html>
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
    <title>Technical Indicators<\/title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"><\/script>
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
    <\/style>
<\/head>
<\/body>
    <div class="container">
        <div class="header">
            <h1>Technical Indicators Analysis<\/h1>
        <\/div>
        
        <div class="indicators-grid">
            <div class="indicator-card">
                <div class="indicator-title">RSI (14)<\/div>
                <div class="indicator-value">{technical_indicators.get('RSI', 0):.1f}<\/div>
                <div class="indicator-bar">
                    <div class="indicator-fill" style="width: {min(100, technical_indicators.get('RSI', 0))}%"><\/div>
                <\/div>
                <div class="indicator-description">
                    {ChartGenerator._get_rsi_description(technical_indicators.get('RSI', 0))}
                <\/div>
            <\/div>
            
            <div class="indicator-card">
                <div class="indicator-title">MACD<\/div>
                <div class="indicator-value">{technical_indicators.get('MACD', 0):.4f}<\/div>
                <div class="indicator-bar">
                    <div class="indicator-fill" style="width: {min(100, max(0, technical_indicators.get('MACD', 0) * 50 + 50))}%"><\/div>
                <\/div>
                <div class="indicator-description">
                    {ChartGenerator._get_macd_description(technical_indicators.get('MACD', 0))}
                <\/div>
            <\/div>
            
            <div class="indicator-card">
                <div class="indicator-title">ADX (14)<\/div>
                <div class="indicator-value">{technical_indicators.get('ADX', 0):.1f}<\/div>
                <div class="indicator-bar">
                    <div class="indicator-fill" style="width: {min(100, technical_indicators.get('ADX', 0))}%"><\/div>
                <\/div>
                <div class="indicator-description">
                    {ChartGenerator._get_adx_description(technical_indicators.get('ADX', 0))}
                <\/div>
            <\/div>
            
            <div class="indicator-card">
                <div class="indicator-title">ATR (14)<\/div>
                <div class="indicator-value">{technical_indicators.get('ATR', 0):.4f}<\/div>
                <div class="indicator-description">Average True Range for volatility assessment<\/div>
            <\/div>
            
            <div class="indicator-card">
                <div class="indicator-title">Volatility<\/div>
                <div class="indicator-value">{technical_indicators.get('Volatility', 0):.4f}<\/div>
                <div class="indicator-bar">
                    <div class="indicator-fill" style="width: {min(100, technical_indicators.get('Volatility', 0) * 1000)}%"><\/div>
                <\/div>
                <div class="indicator-description">Historical price volatility<\/div>
            <\/div>
        <\/div>
    <\/div>
<\/body>
<\/html>
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
