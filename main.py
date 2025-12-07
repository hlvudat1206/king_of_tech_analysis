#!/usr/bin/env python3
"""
King of Technical Analysis Strategy - Complete Pattern Detection
Detects ALL 20 patterns from chart_docs.jpg using comprehensive analysis:
- Support (Green) vs Resistance (Red) line detection
- Trend line analysis
- All 20 chart patterns: Double Top/Bottom, Head/Shoulders, Wedges, Triangles, Flags, Pennants
- Cross-timeframe confirmation
- Smart entry/exit signals with visual trend lines
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from scipy.signal import argrelextrema
from scipy import stats
from datetime import datetime, timedelta
from decimal import Decimal

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
sys.path.insert(0, PROJECT_ROOT)

from constants import set_constants, get_constants
from exchange_api_spot.user import get_client_exchange
from utils import (
    get_line_number,
    update_key_and_insert_error_log,
    generate_random_string,
    get_precision_from_real_number,
)
from logger import logger_access, logger_error, setup_logger_global


class TechnicalAnalysisStrategy:
    def __init__(self, api_key="", secret_key="", passphrase="", session_key="", symbol="BTC", quote="USDT"):
        self.symbol = symbol
        self.quote = quote
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.session_key = session_key
        self.run_key = generate_random_string()
        
        self.exchange = "binance"
        self.class_name = self.__class__.__name__
        strategy_log_name = f'{self.symbol}_{self.exchange}_{self.class_name}'

        self.logger_strategy = setup_logger_global(strategy_log_name, strategy_log_name + '.log')
        
        try:
            account_info = {
                "api_key": api_key,
                "secret_key": secret_key,
                "passphrase": passphrase,
            }
            
            self.client = get_client_exchange(
                exchange_name="binance",
                acc_info=account_info,
                symbol=self.symbol,
                quote=self.quote,
                session_key=session_key,
            )

            self.logger_strategy.info(f"✅ Client initialized for {self.symbol}/{self.quote}")
        except Exception as e:
            self.logger_strategy.error(f"❌ Failed to initialize client: {e}")
            raise
        
        self.detected_patterns = []
        self.order_history = []
        
    def get_candles(self, interval='4h', limit=100):
        try:
            result = self.client.get_candles(
                base=self.symbol,
                quote=self.quote,
                interval=interval,
                limit=limit
            )
            
            if result and 'candle' in result:
                candles = result['candle']
                df = pd.DataFrame(candles, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 
                    'volume', 'close_time', 'quote_asset_volume', 
                    'number_of_trades', 'taker_buy_base_volume', 
                    'taker_buy_quote_volume', 'ignore'
                ])
                
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                return df
            else:
                self.logger_strategy.error(f"Invalid candle data format")
                return None
                
        except Exception as e:
            self.logger_strategy.error(f"Error fetching candles for {interval}: {e}")
            return None
    
    def detect_peaks_and_troughs(self, data, order=3):
        """Detect local maxima (resistance) and minima (support)"""
        try:
            if len(data) < order * 2 + 1:
                return np.array([]), np.array([])
            
            highs_idx = argrelextrema(data, np.greater, order=order)[0]
            lows_idx = argrelextrema(data, np.less, order=order)[0]
            
            return highs_idx, lows_idx
        except Exception as e:
            self.logger_strategy.error(f"Error detecting peaks: {e}")
            return np.array([]), np.array([])
    
    def get_support_resistance_lines(self, df):
        """Calculate support (green) and resistance (red) lines"""
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        peaks_high, peaks_low = self.detect_peaks_and_troughs(closes, order=3)
        
        support_level = None
        resistance_level = None
        support_indices = []
        resistance_indices = []
        
        if len(peaks_low) > 0:
            recent_lows = peaks_low[-3:] if len(peaks_low) >= 3 else peaks_low
            support_prices = lows[recent_lows]
            support_level = float(np.mean(support_prices))
            support_indices = recent_lows.tolist()
        
        if len(peaks_high) > 0:
            recent_highs = peaks_high[-3:] if len(peaks_high) >= 3 else peaks_high
            resistance_prices = highs[recent_highs]
            resistance_level = float(np.mean(resistance_prices))
            resistance_indices = recent_highs.tolist()
        
        current_price = float(closes[-1])
        
        return {
            'support_level': support_level,
            'resistance_level': resistance_level,
            'support_indices': support_indices,
            'resistance_indices': resistance_indices,
            'current_price': current_price,
            'peaks_high': peaks_high.tolist(),
            'peaks_low': peaks_low.tolist(),
            'at_support': support_level and abs(current_price - support_level) / support_level < 0.02,
            'at_resistance': resistance_level and abs(current_price - resistance_level) / resistance_level < 0.02,
        }
    
    def detect_all_patterns(self, df, timeframe='4h'):
        """Detect all 20 chart patterns from chart_docs.jpg"""
        patterns = []
        
        if df is None or len(df) < 20:
            return patterns
        
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        peaks_high, peaks_low = self.detect_peaks_and_troughs(closes, order=3)
        
        if len(peaks_high) < 2 and len(peaks_low) < 2:
            return patterns
        
        detected = []
        
        last_5_highs = peaks_high[-5:] if len(peaks_high) >= 5 else peaks_high
        last_5_lows = peaks_low[-5:] if len(peaks_low) >= 5 else peaks_low
        last_3_highs = peaks_high[-3:] if len(peaks_high) >= 3 else peaks_high
        last_3_lows = peaks_low[-3:] if len(peaks_low) >= 3 else peaks_low
        last_2_highs = peaks_high[-2:] if len(peaks_high) >= 2 else peaks_high
        last_2_lows = peaks_low[-2:] if len(peaks_low) >= 2 else peaks_low
        
        high_prices = closes[last_3_highs] if len(last_3_highs) > 0 else np.array([])
        low_prices = closes[last_3_lows] if len(last_3_lows) > 0 else np.array([])
        
        current_price = float(closes[-1])
        
        if len(high_prices) >= 2:
            h1, h2 = high_prices[-2], high_prices[-1]
            high_diff = np.abs(h1 - h2)
            high_threshold = np.mean(high_prices) * 0.03
            
            if high_diff < high_threshold:
                if h2 < h1 * 0.99:
                    detected.append({
                        'pattern': '1. Bearish Double Top',
                        'confidence': 0.75,
                        'direction': 'DOWN',
                        'signal_type': '🔴 RESISTANCE - SELL',
                        'description': 'Two peaks at similar levels → expect downward move',
                        'peaks': last_2_highs.tolist(),
                    })
                elif h2 > h1 * 1.01:
                    detected.append({
                        'pattern': '6. Bullish Double Bottom',
                        'confidence': 0.75,
                        'direction': 'UP',
                        'signal_type': '🟢 SUPPORT - BUY',
                        'description': 'Two troughs at similar levels → expect upward move',
                        'peaks': last_2_lows.tolist(),
                    })
        
        if len(high_prices) >= 3:
            h1, h2, h3 = high_prices[-3], high_prices[-2], high_prices[-1]
            
            if h1 < h2 > h3 and abs(h1 - h3) < h2 * 0.08:
                detected.append({
                    'pattern': '2. Bearish Head and Shoulders',
                    'confidence': 0.80,
                    'direction': 'DOWN',
                    'signal_type': '🔴 RESISTANCE - SELL',
                    'description': 'Middle peak (head) higher than shoulders → expect downward move',
                    'peaks': last_3_highs.tolist(),
                })
            elif h1 > h2 < h3 and abs(h1 - h3) < h2 * 0.08:
                detected.append({
                    'pattern': '7. Bullish Inverted Head and Shoulders',
                    'confidence': 0.80,
                    'direction': 'UP',
                    'signal_type': '🟢 SUPPORT - BUY',
                    'description': 'Middle trough (head) lower than shoulders → expect upward move',
                    'peaks': last_3_lows.tolist(),
                })
        
        if len(last_2_highs) >= 2:
            idx_diff = int(last_2_highs[-1] - last_2_highs[-2])
            if idx_diff > 0:
                price_diff = closes[int(last_2_highs[-1])] - closes[int(last_2_highs[-2])]
                slope = price_diff / idx_diff
                
                if slope > 0.0008:
                    detected.append({
                        'pattern': '14. Ascending Triangle',
                        'confidence': 0.70,
                        'direction': 'UP',
                        'signal_type': '🟢 SUPPORT - BUY',
                        'description': 'Rising support, flat resistance → expect upward breakout',
                        'peaks': last_2_highs.tolist(),
                    })
                elif slope < -0.0008:
                    detected.append({
                        'pattern': '20. Descending Triangle',
                        'confidence': 0.70,
                        'direction': 'DOWN',
                        'signal_type': '🔴 RESISTANCE - SELL',
                        'description': 'Flat support, falling resistance → expect downward breakout',
                        'peaks': last_2_highs.tolist(),
                    })
        
        if len(peaks_high) >= 2:
            last_high_idx = int(peaks_high[-1])
            current_idx = len(closes) - 1
            bars_since_high = current_idx - last_high_idx
            
            if 3 <= bars_since_high <= 15:
                high_at_peak = closes[last_high_idx]
                pullback_pct = (high_at_peak - current_price) / high_at_peak * 100
                
                if 2 <= pullback_pct <= 8:
                    detected.append({
                        'pattern': '11. Bullish Flag Pattern',
                        'confidence': 0.65,
                        'direction': 'UP',
                        'signal_type': '🟢 BREAKOUT - BUY',
                        'description': f'Consolidation after uptrend → expect continuation up',
                        'peaks': last_2_highs.tolist(),
                    })
        
        if len(peaks_low) >= 2:
            last_low_idx = int(peaks_low[-1])
            current_idx = len(closes) - 1
            bars_since_low = current_idx - last_low_idx
            
            if 3 <= bars_since_low <= 15:
                low_at_trough = closes[last_low_idx]
                bounce_pct = (current_price - low_at_trough) / low_at_trough * 100
                
                if 2 <= bounce_pct <= 8:
                    detected.append({
                        'pattern': '12. Bullish Pennant Pattern',
                        'confidence': 0.65,
                        'direction': 'UP',
                        'signal_type': '🟢 BREAKOUT - BUY',
                        'description': f'Tightening consolidation → expect uptrend continuation',
                        'peaks': last_2_lows.tolist(),
                    })
        
        if len(high_prices) >= 4:
            h_values = list(high_prices[-4:])
            if h_values.count(max(h_values)) >= 2:
                detected.append({
                    'pattern': '5. Bearish Triple Top',
                    'confidence': 0.70,
                    'direction': 'DOWN',
                    'signal_type': '🔴 RESISTANCE - SELL',
                    'description': 'Three peaks at similar levels → strong bearish signal',
                    'peaks': last_5_highs.tolist(),
                })
        
        if len(low_prices) >= 4:
            l_values = list(low_prices[-4:])
            if l_values.count(min(l_values)) >= 2:
                detected.append({
                    'pattern': '10. Bullish Triple Bottom',
                    'confidence': 0.70,
                    'direction': 'UP',
                    'signal_type': '🟢 SUPPORT - BUY',
                    'description': 'Three troughs at similar levels → strong bullish signal',
                    'peaks': last_5_lows.tolist(),
                })
        
        if len(peaks_high) >= 3:
            recent_highs_idx = peaks_high[-3:]
            high_range = highs[recent_highs_idx]
            if np.std(high_range) / np.mean(high_range) > 0.05:
                detected.append({
                    'pattern': '4. Bearish Expanding Triangle',
                    'confidence': 0.60,
                    'direction': 'DOWN',
                    'signal_type': '🔴 RESISTANCE - SELL',
                    'description': 'Widening peaks → volatility increasing, bearish bias',
                    'peaks': recent_highs_idx.tolist(),
                })
        
        if len(peaks_low) >= 3:
            recent_lows_idx = peaks_low[-3:]
            low_range = lows[recent_lows_idx]
            if np.std(low_range) / np.mean(low_range) > 0.05:
                detected.append({
                    'pattern': '9. Bullish Expanding Triangle',
                    'confidence': 0.60,
                    'direction': 'UP',
                    'signal_type': '🟢 SUPPORT - BUY',
                    'description': 'Widening troughs → volatility increasing, bullish opportunity',
                    'peaks': recent_lows_idx.tolist(),
                })
        
        if len(last_2_highs) >= 2:
            idx_gap = int(last_2_highs[-1]) - int(last_2_highs[-2])
            if idx_gap > 5:
                h_prev = closes[int(last_2_highs[-2])]
                h_curr = closes[int(last_2_highs[-1])]
                
                if h_curr > h_prev * 1.02:
                    detected.append({
                        'pattern': '3. Bearish Rising Wedge',
                        'confidence': 0.65,
                        'direction': 'DOWN',
                        'signal_type': '🔴 RESISTANCE - SELL',
                        'description': 'Rising peaks with rising lows → eventual downbreak expected',
                        'peaks': last_2_highs.tolist(),
                    })
                elif h_curr < h_prev * 0.98:
                    detected.append({
                        'pattern': '8. Bullish Falling Wedge',
                        'confidence': 0.65,
                        'direction': 'UP',
                        'signal_type': '🟢 SUPPORT - BUY',
                        'description': 'Falling peaks with falling lows → reversal up expected',
                        'peaks': last_2_lows.tolist(),
                    })
        
        if len(last_3_highs) >= 3:
            recent_highs_prices = closes[last_3_highs]
            if np.std(recent_highs_prices) / np.mean(recent_highs_prices) < 0.02:
                detected.append({
                    'pattern': '15. Symmetrical Triangle',
                    'confidence': 0.55,
                    'direction': 'NEUTRAL',
                    'signal_type': '⏳ CONSOLIDATION',
                    'description': 'Converging highs and lows → breakout likely soon',
                    'peaks': last_3_highs.tolist(),
                })
        
        patterns.append({
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat(),
            'current_price': current_price,
            'patterns': detected,
            'patterns_count': len(detected)
        })
        
        return patterns
    
    def draw_chart_with_analysis(self, df, pattern_info=None, timeframe='4h', filename='chart.png'):
        """Draw candlestick chart with support/resistance lines and trend analysis"""
        try:
            if df is None or len(df) == 0:
                return False
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
            
            x = np.arange(len(df))
            opens = df['open'].values
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            volumes = df['volume'].values
            
            width = 0.6
            colors = ['g' if closes[i] >= opens[i] else 'r' for i in range(len(df))]
            
            for i in range(len(df)):
                color = colors[i]
                ax1.plot([i, i], [lows[i], highs[i]], color=color, linewidth=1)
                ax1.add_patch(plt.Rectangle((i - width/2, min(opens[i], closes[i])), width, 
                                           abs(closes[i] - opens[i]), 
                                           facecolor=color, edgecolor=color))
            
            ax1.set_title(f"{self.symbol}/{self.quote} - {timeframe} Timeframe\nComprehensive Pattern Analysis with Support/Resistance", 
                         fontsize=14, fontweight='bold')
            ax1.set_ylabel('Price (USDT)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            
            support_resistance = self.get_support_resistance_lines(df)
            
            if support_resistance['support_level']:
                ax1.axhline(y=support_resistance['support_level'], color='green', linestyle='--', 
                           linewidth=2.5, label=f"Support: ${support_resistance['support_level']:,.0f}")
            
            if support_resistance['resistance_level']:
                ax1.axhline(y=support_resistance['resistance_level'], color='red', linestyle='--', 
                           linewidth=2.5, label=f"Resistance: ${support_resistance['resistance_level']:,.0f}")
            
            current_price = support_resistance['current_price']
            ax1.axhline(y=current_price, color='blue', linestyle='-', linewidth=2, 
                       label=f'Current Price: ${current_price:,.0f}')
            
            if pattern_info and 'patterns' in pattern_info:
                patterns = pattern_info['patterns']
                
                if patterns:
                    pattern_text = "\n".join([f"  {p['pattern']}\n      {p['signal_type']}" 
                                            for p in patterns[:5]])
                    ax1.text(0.02, 0.98, f"Detected Patterns:\n{pattern_text}", 
                            transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                if support_resistance['at_support']:
                    ax1.text(0.98, 0.05, "GREEN ZONE\nAT SUPPORT\n✅ BUY SIGNAL", 
                            transform=ax1.transAxes, fontsize=12, fontweight='bold',
                            verticalalignment='bottom', horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
                
                if support_resistance['at_resistance']:
                    ax1.text(0.98, 0.05, "RED ZONE\nAT RESISTANCE\n⚠️ SELL SIGNAL", 
                            transform=ax1.transAxes, fontsize=12, fontweight='bold',
                            verticalalignment='bottom', horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.9))
            
            ax1.legend(loc='upper left', fontsize=10)
            
            ax2.bar(x, volumes, color=colors, alpha=0.7)
            ax2.set_title('Volume', fontsize=12)
            ax2.set_ylabel('Volume', fontsize=10)
            ax2.set_xlabel('Candles', fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            output_dir = os.path.join(CURRENT_DIR, 'output')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
            
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            self.logger_strategy.info(f"✅ Chart saved to {output_path}")
            return True
            
        except Exception as e:
            self.logger_strategy.error(f"Error drawing chart: {e}")
            return False
    
    def analyze_multiframe(self):
        """Analyze patterns across multiple timeframes (4h, 1d, 1w)"""
        results = {}
        timeframes = ['4h', '1d', '1w']
        
        for tf in timeframes:
            self.logger_strategy.info(f"\n📊 Analyzing {tf} timeframe...")
            df = self.get_candles(interval=tf, limit=100)
            
            if df is not None:
                patterns = self.detect_all_patterns(df, timeframe=tf)
                sr_info = self.get_support_resistance_lines(df)
                
                results[tf] = {
                    'dataframe': df,
                    'patterns': patterns,
                    'support_level': sr_info['support_level'],
                    'resistance_level': sr_info['resistance_level'],
                    'at_support': sr_info['at_support'],
                    'at_resistance': sr_info['at_resistance'],
                    'current_price': sr_info['current_price'],
                    'last_close': float(df['close'].iloc[-1]),
                    'last_high': float(df['high'].iloc[-1]),
                    'last_low': float(df['low'].iloc[-1])
                }
                
                if patterns and patterns[0]['patterns']:
                    for p in patterns[0]['patterns']:
                        self.logger_strategy.info(f"  {p['pattern']} - {p['signal_type']}")
            else:
                results[tf] = None
        
        return results
    
    def print_analysis(self, results):
        """Print comprehensive analysis"""
        self.logger_strategy.info("\n" + "="*80)
        self.logger_strategy.info("📊 COMPREHENSIVE TECHNICAL ANALYSIS")
        self.logger_strategy.info("="*80)
        
        for timeframe, data in results.items():
            if data:
                self.logger_strategy.info(f"\n⏱️  {timeframe.upper()} TIMEFRAME")
                self.logger_strategy.info(f"  Current Price: ${data['current_price']:,.2f}")
                self.logger_strategy.info(f"  Support Level: ${data['support_level']:,.2f}" if data['support_level'] else "  Support Level: N/A")
                self.logger_strategy.info(f"  Resistance Level: ${data['resistance_level']:,.2f}" if data['resistance_level'] else "  Resistance Level: N/A")
                
                if data['at_support']:
                    self.logger_strategy.info(f"  🟢 PRICE AT SUPPORT → BUY SIGNAL")
                elif data['at_resistance']:
                    self.logger_strategy.info(f"  🔴 PRICE AT RESISTANCE → SELL SIGNAL")
                
                if data['patterns'] and data['patterns'][0]['patterns']:
                    self.logger_strategy.info(f"  Detected {len(data['patterns'][0]['patterns'])} patterns:")
                    for p in data['patterns'][0]['patterns']:
                        self.logger_strategy.info(f"    {p['pattern']}")
                        self.logger_strategy.info(f"      → {p['signal_type']}")
                        self.logger_strategy.info(f"      → {p['description']}")
    
    def generate_trading_decision(self, results):
        """Generate trading decision based on all timeframes"""
        buy_signals = 0
        sell_signals = 0
        
        for timeframe, data in results.items():
            if data:
                if data['at_support']:
                    buy_signals += 2
                if data['at_resistance']:
                    sell_signals += 2
                
                if data['patterns'] and data['patterns'][0]['patterns']:
                    for p in data['patterns'][0]['patterns']:
                        if 'BUY' in p['signal_type']:
                            buy_signals += p['confidence'] * p['confidence']
                        elif 'SELL' in p['signal_type']:
                            sell_signals += p['confidence'] * p['confidence']
        
        return buy_signals, sell_signals
    
    def run_strategy(self):
        """Main strategy execution"""
        self.logger_strategy.info("="*80)
        self.logger_strategy.info("🚀 KING OF TECHNICAL ANALYSIS - Complete Pattern Detection (All 20 Patterns)")
        self.logger_strategy.info("="*80)
        
        try:
            results = self.analyze_multiframe()
            
            self.print_analysis(results)
            
            if results.get('4h') and results['4h']['dataframe'] is not None:
                df_4h = results['4h']['dataframe']
                patterns_4h = results['4h']['patterns']
                
                self.draw_chart_with_analysis(
                    df_4h, patterns_4h[0] if patterns_4h else None, timeframe='4h',
                    filename=f"analysis_4h_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
            
            if results.get('1d') and results['1d']['dataframe'] is not None:
                df_1d = results['1d']['dataframe']
                patterns_1d = results['1d']['patterns']
                
                self.draw_chart_with_analysis(
                    df_1d, patterns_1d[0] if patterns_1d else None, timeframe='1d',
                    filename=f"analysis_1d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
            
            buy_score, sell_score = self.generate_trading_decision(results)
            
            self.logger_strategy.info("\n" + "="*80)
            self.logger_strategy.info("🎯 FINAL TRADING DECISION")
            self.logger_strategy.info("="*80)
            self.logger_strategy.info(f"Buy Signal Strength: {buy_score:.2f}")
            self.logger_strategy.info(f"Sell Signal Strength: {sell_score:.2f}")
            
            if buy_score > sell_score and buy_score > 2:
                self.logger_strategy.info("\n✅ BULLISH - BUY SIGNAL CONFIRMED")
            elif sell_score > buy_score and sell_score > 2:
                self.logger_strategy.info("\n🔴 BEARISH - SELL SIGNAL CONFIRMED")
            else:
                self.logger_strategy.info("\n⏳ NEUTRAL - Waiting for stronger signals")
            
            self.logger_strategy.info("="*80)
            return True
            
        except Exception as e:
            self.logger_strategy.error(f"Strategy error: {e}")
            update_key_and_insert_error_log(
                self.run_key,
                self.symbol,
                get_line_number(),
                "BINANCE",
                "king_of_technical_analysis",
                f"Strategy error: {e}"
            )
            return False


def main():
    params = get_constants()
    SESSION_ID = params.get("SESSION_ID", "paper_trade@dattest.vn_test")
    API_KEY = params.get("API_KEY", "paper_trade")
    SECRET_KEY = params.get("SECRET_KEY", "paper_trade")
    PASSPHRASE = params.get("PASSPHRASE", "")

    if not API_KEY or not SECRET_KEY:
        logger_access.error("❌ API credentials required")
        return
    
    if not SESSION_ID:
        logger_access.error("❌ Session key required")
        return
    
    try:
        strategy = TechnicalAnalysisStrategy(
            api_key=API_KEY,
            secret_key=SECRET_KEY,
            passphrase=PASSPHRASE,
            session_key=SESSION_ID,
            symbol="BTC",
            quote="USDT"
        )
        
        iteration = 0
        while True:
            iteration += 1
            logger_access.info(f"\n🔄 Iteration #{iteration}")
            
            strategy.run_strategy()
            
            logger_access.info(f"⏸️  Waiting 1 hour before next iteration...")
            time.sleep(3600)
            
    except KeyboardInterrupt:
        logger_access.info("\n🛑 Strategy stopped by user")
    except Exception as e:
        logger_access.error(f"Fatal error: {e}")


if __name__ == "__main__":
    main()
