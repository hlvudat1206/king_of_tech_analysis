#!/usr/bin/env python3
"""
King of Technical Analysis Strategy
Detects chart patterns using peak detection and multi-timeframe analysis.
Identifies patterns like Double Top, Head and Shoulders, Wedges, Triangles, etc.
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
    
    def detect_peaks(self, data, order=5):
        """Detect local maxima and minima in price data"""
        try:
            if len(data) < order * 2 + 1:
                return [], []
            
            highs_idx = argrelextrema(data, np.greater, order=order)[0]
            lows_idx = argrelextrema(data, np.less, order=order)[0]
            
            return highs_idx, lows_idx
        except Exception as e:
            self.logger_strategy.error(f"Error detecting peaks: {e}")
            return [], []
    
    def detect_patterns(self, df):
        """Detect technical analysis patterns from the dataframe"""
        patterns = []
        
        if df is None or len(df) < 10:
            return patterns
        
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        
        peaks_high, peaks_low = self.detect_peaks(closes, order=3)
        
        if len(peaks_high) >= 3:
            recent_peaks_h = peaks_high[-3:]
            recent_lows = peaks_low[-3:] if len(peaks_low) >= 3 else []
            
            pattern_info = {
                'timestamp': datetime.now().isoformat(),
                'recent_peaks_high': recent_peaks_h.tolist(),
                'recent_peaks_low': recent_lows.tolist(),
                'prices_at_high_peaks': closes[recent_peaks_h].tolist() if len(recent_peaks_h) > 0 else [],
                'patterns': self._analyze_pattern(closes, highs, lows, peaks_high, peaks_low)
            }
            patterns.append(pattern_info)
        
        return patterns
    
    def _analyze_pattern(self, closes, highs, lows, peaks_high, peaks_low):
        """Analyze specific chart patterns"""
        detected = []
        
        if len(peaks_high) < 2:
            return detected
        
        last_3_highs = peaks_high[-3:] if len(peaks_high) >= 3 else peaks_high[-2:]
        last_3_lows = peaks_low[-3:] if len(peaks_low) >= 3 else peaks_low[-2:]
        
        high_prices = closes[last_3_highs]
        low_prices = closes[last_3_lows]
        
        if len(high_prices) >= 2:
            high_diff = np.abs(high_prices[0] - high_prices[-1])
            high_threshold = np.mean(high_prices) * 0.02
            
            if high_diff < high_threshold and high_prices[-1] < high_prices[0]:
                detected.append({
                    'pattern': 'Double Top (Bearish)',
                    'confidence': 0.7,
                    'direction': 'DOWN',
                    'peaks': last_3_highs.tolist()
                })
            elif high_diff < high_threshold and high_prices[-1] > high_prices[0]:
                detected.append({
                    'pattern': 'Double Bottom (Bullish)',
                    'confidence': 0.7,
                    'direction': 'UP',
                    'peaks': last_3_highs.tolist()
                })
        
        if len(high_prices) >= 3:
            h1, h2, h3 = high_prices[0], high_prices[1], high_prices[2]
            
            if h1 < h2 > h3 and abs(h1 - h3) < h2 * 0.05:
                detected.append({
                    'pattern': 'Head and Shoulders (Bearish)',
                    'confidence': 0.75,
                    'direction': 'DOWN',
                    'peaks': last_3_highs.tolist()
                })
            elif h1 > h2 < h3 and abs(h1 - h3) < h2 * 0.05:
                detected.append({
                    'pattern': 'Inverse Head and Shoulders (Bullish)',
                    'confidence': 0.75,
                    'direction': 'UP',
                    'peaks': last_3_highs.tolist()
                })
        
        if len(last_3_highs) >= 2:
            indices_diff = last_3_highs[-1] - last_3_highs[-2]
            price_diff = closes[last_3_highs[-1]] - closes[last_3_highs[-2]]
            
            if indices_diff > 0:
                slope = price_diff / indices_diff
                
                if slope > 0.0005:
                    detected.append({
                        'pattern': 'Ascending Triangle (Bullish)',
                        'confidence': 0.65,
                        'direction': 'UP',
                        'peaks': last_3_highs.tolist()
                    })
                elif slope < -0.0005:
                    detected.append({
                        'pattern': 'Descending Triangle (Bearish)',
                        'confidence': 0.65,
                        'direction': 'DOWN',
                        'peaks': last_3_highs.tolist()
                    })
        
        return detected
    
    def draw_candles(self, df, pattern_info=None, filename='candles.png'):
        """Draw candlestick chart with detected peaks"""
        try:
            self.logger_strategy.info(f"Chart Testing to 1")

            if df is None or len(df) == 0:
                return False
            
            df_plot = df.copy()
            df_plot['Date'] = df_plot['open_time']
            df_plot = df_plot.set_index('Date')
            
            ohlc_data = df_plot[['open', 'high', 'low', 'close', 'volume']].astype(float)
            
            self.logger_strategy.info(f"Chart Testing to 2")

            apds = []
            
            if pattern_info and 'recent_peaks_high' in pattern_info:
                self.logger_strategy.info(f"Chart Testing to 3")

                peaks_high = pattern_info['recent_peaks_high']
                if len(peaks_high) > 0:
                    scatter_data = [df['close'].iloc[p] if p in peaks_high and p < len(df) else np.nan for p in range(len(df))]
                    scatter_high = mpf.make_addplot(
                        scatter_data,
                        type='scatter',
                        marker='^',
                        markersize=100,
                        color='red'
                    )
                    apds.append(scatter_high)
            
            kwargs = dict(type='candle', volume=True, style='charles')
            self.logger_strategy.info(f"Chart Testing to 4")

            if apds:
                kwargs['addplot'] = apds
            
            output_dir = os.path.join(CURRENT_DIR, 'output')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
            self.logger_strategy.info(f"Chart Testing to 5")

            mpf.plot(ohlc_data, **kwargs, savefig=output_path)
            self.logger_strategy.info(f"✅ Chart saved to {output_path}")
            return True
            
        except Exception as e:
            self.logger_strategy.error(f"Error drawing candles: {e}")
            return False
    
    def analyze_multiframe(self):
        """Analyze patterns across multiple timeframes (4h, 1d, 1w)"""
        results = {}
        timeframes = ['4h', '1d', '1w']
        
        for tf in timeframes:
            self.logger_strategy.info(f"\n📊 Analyzing {tf} timeframe...")
            df = self.get_candles(interval=tf, limit=100)
            
            if df is not None:
                patterns = self.detect_patterns(df)
                results[tf] = {
                    'dataframe': df,
                    'patterns': patterns,
                    'last_close': float(df['close'].iloc[-1]),
                    'last_high': float(df['high'].iloc[-1]),
                    'last_low': float(df['low'].iloc[-1])
                }
                
                if patterns and patterns[0]['patterns']:
                    for p in patterns[0]['patterns']:
                        self.logger_strategy.info(f"  🎯 Pattern: {p['pattern']} - Confidence: {p['confidence']:.0%}")
            else:
                results[tf] = None
        
        return results
    
    def print_detected_pattern(self, results):
        """Print detected patterns in a readable format"""
        self.logger_strategy.info("\n" + "="*60)
        self.logger_strategy.info("🔍 DETECTED PATTERNS ANALYSIS")
        self.logger_strategy.info("="*60)
        
        for timeframe, data in results.items():
            if data and data['patterns']:
                self.logger_strategy.info(f"\n⏱️ Timeframe: {timeframe}")
                self.logger_strategy.info(f"  Last Close: ${data['last_close']:,.2f}")
                self.logger_strategy.info(f"  Last High:  ${data['last_high']:,.2f}")
                self.logger_strategy.info(f"  Last Low:   ${data['last_low']:,.2f}")
                
                for pattern_info in data['patterns']:
                    if pattern_info['patterns']:
                        for pattern in pattern_info['patterns']:
                            self.logger_strategy.info(f"\n  📈 Pattern Found: {pattern['pattern']}")
                            self.logger_strategy.info(f"     Direction: {pattern['direction']}")
                            self.logger_strategy.info(f"     Confidence: {pattern['confidence']:.0%}")
            else:
                self.logger_strategy.info(f"\n⏱️ Timeframe: {timeframe} - No clear patterns")
    
    def should_place_order(self, results):
        """Determine if order should be placed based on multi-timeframe analysis"""
        bullish_count = 0
        bearish_count = 0
        
        for timeframe, data in results.items():
            if data and data['patterns']:
                for pattern_info in data['patterns']:
                    for pattern in pattern_info['patterns']:
                        if pattern['direction'] == 'UP':
                            bullish_count += pattern['confidence']
                        elif pattern['direction'] == 'DOWN':
                            bearish_count += pattern['confidence']
        
        return bullish_count, bearish_count
    
    def place_order(self, side, quantity, current_price):
        """Place a market order for spot trading"""
        try:
            self.logger_strategy.info(f"\n🛒 Placing {side} order...")
            self.logger_strategy.info(f"   Quantity: {quantity} {self.symbol}")
            self.logger_strategy.info(f"   Price: ${current_price:,.2f}")
            
            order_result = self.client.place_order(
                side_order=side,
                quantity=quantity,
                order_type='MARKET'
            )
            
            if order_result and order_result.get('code') == 0:
                order_data = order_result.get('data', {})
                order_id = order_data.get('orderId', 'N/A')
                self.logger_strategy.info(f"✅ Order placed successfully!")
                self.logger_strategy.info(f"   Order ID: {order_id}")
                
                self.order_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'side': side,
                    'quantity': quantity,
                    'price': current_price,
                    'order_id': order_id
                })
                return True
            else:
                self.logger_strategy.error(f"❌ Order failed: {order_result}")
                return False
                
        except Exception as e:
            self.logger_strategy.error(f"Error placing order: {e}")
            return False
    
    def run_strategy(self):
        """Main strategy execution"""
        self.logger_strategy.info("="*60)
        self.logger_strategy.info("🚀 King of Technical Analysis Strategy Starting...")
        self.logger_strategy.info("="*60)
        
        try:
            results = self.analyze_multiframe()
            
            self.print_detected_pattern(results)
            
            if results.get('4h') and results['4h']['dataframe'] is not None:
                df_4h = results['4h']['dataframe']
                patterns_4h = results['4h']['patterns']
                
                if patterns_4h:
                    self.draw_candles(df_4h, patterns_4h[0], f"candles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            
            bullish_score, bearish_score = self.should_place_order(results)
            
            self.logger_strategy.info("\n" + "="*60)
            self.logger_strategy.info("📊 TRADING DECISION")
            self.logger_strategy.info("="*60)
            self.logger_strategy.info(f"Bullish Signal Strength: {bullish_score:.2f}")
            self.logger_strategy.info(f"Bearish Signal Strength: {bearish_score:.2f}")
            
            if bullish_score > bearish_score and bullish_score > 1.0:
                self.logger_strategy.info("\n✅ BULLISH SIGNAL - Considering BUY order")
                
                current_price = results['4h']['last_close'] if results.get('4h') else None
                if current_price:
                    test_quantity = 0.001
                    self.place_order('BUY', test_quantity, current_price)
                    
            elif bearish_score > bullish_score and bearish_score > 1.0:
                self.logger_strategy.info("\n⚠️ BEARISH SIGNAL - Considering SELL order")
                
                current_price = results['4h']['last_close'] if results.get('4h') else None
                if current_price:
                    test_quantity = 0.001
                    self.place_order('SELL', test_quantity, current_price)
            else:
                self.logger_strategy.info("\n⏳ NEUTRAL - Waiting for stronger signals")
            
            self.logger_strategy.info("="*60)
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
            
            logger_access.info(f"⏸️ Waiting 1 hour before next iteration...")
            time.sleep(3600)
            
    except KeyboardInterrupt:
        logger_access.info("\n🛑 Strategy stopped by user")
    except Exception as e:
        logger_access.error(f"Fatal error: {e}")


if __name__ == "__main__":
    main()
