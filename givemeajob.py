import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
import pandas as pd
import logging
import time
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrendFollowingBot:
    def __init__(self, trailing_stop_percent=0.05):
        load_dotenv()
        self.api_key = os.getenv('APCA_API_KEY_ID')
        self.api_secret = os.getenv('APCA_API_SECRET_KEY')
        self.trailing_stop_percent = trailing_stop_percent

        if not all([self.api_key, self.api_secret]):
            raise ValueError("API credentials not found in environment variables")

        self.api = tradeapi.REST(
            self.api_key,
            self.api_secret,
            base_url='https://paper-api.alpaca.markets'
        )
        self.validate_api_connection()

    def validate_api_connection(self):
        try:
            account = self.api.get_account()
            logger.info(f"Connected to Alpaca. Account status: {account.status}")
            return account
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca API: {e}")
            raise

    def get_historical_data(self, symbol, limit=100):
        try:
            end = datetime.now()
            start = end - timedelta(days=limit)
            bars = self.api.get_bars(symbol, TimeFrame.Day, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), limit=limit)
            df = pd.DataFrame([bar._raw for bar in bars]).sort_values('timestamp').reset_index(drop=True)
            return df if not df.empty else None
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None

    def calculate_rsi(self, data, period=14):
        delta = data['close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        data['RSI'] = rsi
        return data

    def check_entry_signal(self, data, rsi_threshold=50):
        self.calculate_rsi(data)
        current_rsi = data['RSI'].iloc[-1]
        return current_rsi > rsi_threshold

    def place_trailing_stop_order(self, symbol, qty):
        try:
            latest_trade = self.api.get_latest_trade(symbol)
            current_price = latest_trade.price
            stop_price = current_price * (1 - self.trailing_stop_percent)

            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='gtc'
            )

            trailing_stop_order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                type='trailing_stop',
                trail_percent=self.trailing_stop_percent * 100,
                time_in_force='gtc'
            )

            logger.info(f"Trailing stop order placed: Buy {qty} shares of {symbol}, trailing stop at {self.trailing_stop_percent*100}%")
            return order, trailing_stop_order

        except Exception as e:
            logger.error(f"Error placing trailing stop order: {e}")
            return None

    def run_strategy(self, symbol, check_interval=900):
        logger.info(f"Starting trend-following bot for {symbol} with a {self.trailing_stop_percent*100}% trailing stop")

        while True:
            try:
                clock = self.api.get_clock()
                if not clock.is_open:
                    next_open = clock.next_open.strftime('%Y-%m-%d %H:%M:%S')
                    logger.info(f"Market is closed. Next opening at {next_open}")
                    time.sleep(min(check_interval, (clock.next_open - clock.timestamp).seconds))
                    continue

                data = self.get_historical_data(symbol)
                if data is None or len(data) < 20:
                    logger.warning("Not enough data for analysis; retrying in 5 minutes.")
                    time.sleep(300)
                    continue

                if self.check_entry_signal(data):
                    logger.info("Entry signal detected based on RSI")
                    self.place_trailing_stop_order(symbol, qty=1)

                else:
                    logger.info("No entry signal; waiting for the next interval.")

                time.sleep(check_interval)

            except KeyboardInterrupt:
                logger.info("Stopping bot...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                logger.error("Retrying in 5 minutes...")
                time.sleep(300)

if __name__ == "__main__":
    try:
        bot = TrendFollowingBot(trailing_stop_percent=0.05)  # Set trailing stop to 5%
        bot.run_strategy("AAPL")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
