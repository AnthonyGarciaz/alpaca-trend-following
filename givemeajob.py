import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
import pandas as pd
import logging
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Configure logging with timezone
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S %Z'
)
logger = logging.getLogger(__name__)


class TrendFollowingBot:
    def __init__(self, trailing_stop_percent=0.05, order_wait_timeout=60):
        load_dotenv()
        self.api_key = os.getenv('APCA_API_KEY_ID')
        self.api_secret = os.getenv('APCA_API_SECRET_KEY')
        self.trailing_stop_percent = trailing_stop_percent
        self.order_wait_timeout = order_wait_timeout
        self.active_positions = {}

        # Initialize timezone (US/Eastern for market hours)
        self.timezone = ZoneInfo("America/New_York")

        if not all([self.api_key, self.api_secret]):
            raise ValueError("API credentials not found in environment variables")

        self.api = tradeapi.REST(
            self.api_key,
            self.api_secret,
            base_url='https://paper-api.alpaca.markets'
        )
        self.validate_api_connection()

    def validate_api_connection(self):
        """Validate API connection and log account information"""
        try:
            account = self.api.get_account()
            logger.info(f"Connected to Alpaca. Account status: {account.status}")
            logger.info(f"Buying power: ${float(account.buying_power):.2f}")
            return account
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca API: {e}")
            raise

    def get_current_market_time(self):
        """Get current market time in US/Eastern"""
        current_time = datetime.now(self.timezone)
        logger.info(f"Current market time: {current_time}")
        return current_time

    def get_latest_price(self, symbol):
        """Get latest price for a symbol with proper error handling"""
        try:
            latest_trade = self.api.get_latest_trade(symbol)
            return float(latest_trade.price)
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            return None

    def get_historical_data(self, symbol, limit=100):
        """Get historical data ensuring we only request actual historical dates"""
        try:
            current_time = self.get_current_market_time()

            # Calculate end date as the last completed trading day
            end = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
            if current_time.hour < 16:  # If before market close, use previous day
                end -= timedelta(days=1)

            # Calculate start date
            start = end - timedelta(days=limit)

            logger.info(f"Requesting historical data from {start} to {end}")

            # Get historical data
            bars = self.api.get_bars(
                symbol,
                TimeFrame.Day,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                limit=limit,
                adjustment='raw'  # Get raw prices without adjustments
            ).df

            if bars.empty:
                logger.warning(f"No data received for {symbol}")
                return None

            # Reset index to make timestamp a column
            bars = bars.reset_index()

            # Convert timestamps to Eastern time
            bars['timestamp'] = pd.to_datetime(bars['timestamp']).dt.tz_convert(self.timezone)

            # Sort by timestamp and get the latest limit days
            bars = bars.sort_values('timestamp').tail(limit).reset_index(drop=True)

            latest_price = self.get_latest_price(symbol)
            logger.info(f"""Historical Data Summary:
- Retrieved {len(bars)} bars
- Date Range: {bars['timestamp'].iloc[0]} to {bars['timestamp'].iloc[-1]}
- Latest Market Price: ${latest_price:.2f}""")

            return bars

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return None

    def calculate_rsi(self, data, period=14):
        """Calculate RSI and additional technical indicators"""
        try:
            if data is None or len(data) < period:
                logger.warning(f"Insufficient data for RSI calculation. Need at least {period} periods.")
                return None

            delta = data['close'].diff(1)
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            avg_gain = gain.rolling(window=period, min_periods=period).mean()
            avg_loss = loss.rolling(window=period, min_periods=period).mean()

            # Handle division by zero
            avg_loss = avg_loss.replace(0, 0.000001)

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            # Add moving average of RSI for trend confirmation
            data['RSI'] = rsi
            data['RSI_MA'] = data['RSI'].rolling(window=10).mean()

            # Calculate ROC (Rate of Change) for additional momentum confirmation
            data['ROC'] = data['close'].pct_change(5) * 100

            latest_price = self.get_latest_price(data.index[-1])
            logger.info(f"""Technical Indicators:
- RSI: {rsi.iloc[-1]:.2f}
- RSI MA: {data['RSI_MA'].iloc[-1]:.2f}
- 5-day ROC: {data['ROC'].iloc[-1]:.2f}%
- Current Price: ${latest_price:.2f}""")

            return data

        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return None

    def check_entry_signal(self, data, rsi_threshold=50):
        """Check if entry conditions are met"""
        try:
            if data is None:
                logger.warning("No data available for entry signal check")
                return False

            data = self.calculate_rsi(data)
            if data is None:
                return False

            current_rsi = data['RSI'].iloc[-1]
            current_rsi_ma = data['RSI_MA'].iloc[-1]
            prev_rsi = data['RSI'].iloc[-2]
            current_roc = data['ROC'].iloc[-1]

            # Enhanced entry conditions
            rsi_strong = current_rsi > rsi_threshold
            rsi_trending_up = current_rsi > current_rsi_ma
            momentum_increasing = current_rsi > prev_rsi
            positive_momentum = current_roc > 0

            entry_signal = (rsi_strong and rsi_trending_up and
                            momentum_increasing and positive_momentum)

            logger.info(f"""Entry Signal Analysis:
RSI > Threshold: {rsi_strong} ({current_rsi:.2f} vs {rsi_threshold})
RSI > MA: {rsi_trending_up} ({current_rsi:.2f} vs {current_rsi_ma:.2f})
RSI Increasing: {momentum_increasing} ({prev_rsi:.2f} to {current_rsi:.2f})
ROC Positive: {positive_momentum} ({current_roc:.2f}%)
Final Signal: {"YES" if entry_signal else "NO"}""")

            return entry_signal

        except Exception as e:
            logger.error(f"Error checking entry signal: {str(e)}")
            return False

    def check_position(self, symbol):
        """Check if we have an existing position with current market value"""
        try:
            position = self.api.get_position(symbol)
            current_price = self.get_latest_price(symbol)
            qty = float(position.qty)
            market_value = qty * current_price if current_price else float(position.market_value)
            return qty, market_value
        except:
            return 0, 0

    def calculate_position_size(self, symbol):
        """Calculate appropriate position size based on account value and risk"""
        try:
            account = self.api.get_account()
            buying_power = float(account.buying_power)
            current_price = self.get_latest_price(symbol)

            if not current_price:
                logger.error("Unable to get current price for position sizing")
                return 1

            # Use 5% of buying power per trade
            max_position_value = buying_power * 0.05
            position_size = int(max_position_value / current_price)

            logger.info(f"""Position Size Calculation:
- Buying Power: ${buying_power:.2f}
- Current Price: ${current_price:.2f}
- Max Position Value: ${max_position_value:.2f}
- Calculated Size: {position_size} shares""")

            return max(1, position_size)  # Minimum 1 share
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 1

    def wait_for_order_fill(self, order_id, timeout=None):
        """Wait for an order to be filled, with timeout"""
        if timeout is None:
            timeout = self.order_wait_timeout

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                order = self.api.get_order(order_id)
                if order.status == 'filled':
                    return order
                elif order.status in ['canceled', 'expired', 'rejected']:
                    logger.error(f"Order failed with status: {order.status}")
                    return None
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error checking order status: {e}")
                return None

        logger.error(f"Order fill timeout after {timeout} seconds")
        return None

    def place_trailing_stop_order(self, symbol, qty):
        """Place a market buy order followed by a trailing stop sell order"""
        try:
            # Check existing position
            current_qty, current_value = self.check_position(symbol)
            if current_qty > 0:
                logger.info(f"Already have position of {current_qty} shares in {symbol}")
                return None

            # Calculate position size and verify current price
            qty = self.calculate_position_size(symbol)
            current_price = self.get_latest_price(symbol)

            if not current_price:
                logger.error("Unable to get current price for order placement")
                return None

            # Place entry order
            entry_order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='gtc'
            )

            logger.info(f"Entry order placed: {entry_order.id}")

            # Wait for entry order to fill
            filled_entry = self.wait_for_order_fill(entry_order.id)
            if not filled_entry:
                logger.error("Entry order failed to fill")
                return None

            filled_price = float(filled_entry.filled_avg_price)
            logger.info(f"Entry order filled at ${filled_price:.2f}")

            # Place trailing stop order
            trailing_stop_order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                type='trailing_stop',
                trail_percent=self.trailing_stop_percent * 100,
                time_in_force='gtc'
            )

            order_value = filled_price * qty
            stop_distance = filled_price * self.trailing_stop_percent

            logger.info(f"""Order Summary:
- Symbol: {symbol}
- Quantity: {qty} shares
- Entry Price: ${filled_price:.2f}
- Total Value: ${order_value:.2f}
- Trailing Stop: {self.trailing_stop_percent * 100}%
- Stop Distance: ${stop_distance:.2f}""")

            # Track the position
            self.active_positions[symbol] = {
                'qty': qty,
                'entry_price': filled_price,
                'entry_time': self.get_current_market_time(),
                'stop_percent': self.trailing_stop_percent
            }

            return filled_entry, trailing_stop_order

        except Exception as e:
            logger.error(f"Error placing orders: {str(e)}")
            return None

    def run_strategy(self, symbol, check_interval=300):
        """Main strategy loop"""
        logger.info(f"""
Starting Trading Bot
-------------------
Symbol: {symbol}
Strategy: RSI Trend Following
Trailing Stop: {self.trailing_stop_percent * 100}%
Check Interval: {check_interval} seconds
Market Time Zone: {self.timezone}
""")

        while True:
            try:
                # Get current market time
                current_time = self.get_current_market_time()

                # Check market hours
                clock = self.api.get_clock()
                if not clock.is_open:
                    next_open = clock.next_open.astimezone(self.timezone)
                    logger.info(f"Market is closed. Next opening at {next_open}")
                    sleep_time = min(check_interval, (next_open - current_time).seconds)
                    time.sleep(sleep_time)
                    continue

                # Get and validate data
                data = self.get_historical_data(symbol)
                if data is None or len(data) < 20:
                    logger.warning("Insufficient data for analysis")
                    time.sleep(300)
                    continue

                # Get current market price
                current_price = self.get_latest_price(symbol)
                if not current_price:
                    logger.error("Unable to get current price")
                    time.sleep(60)
                    continue

                # Check current position
                current_qty, current_value = self.check_position(symbol)

                if current_qty == 0:
                    # Look for entry signals if no position
                    if self.check_entry_signal(data):
                        logger.info("Entry signal confirmed - Placing orders")
                        self.place_trailing_stop_order(symbol, qty=1)
                    else:
                        logger.info("No entry signal detected - Waiting for next check")
                else:
                    # Log position status if we have one
                    position_info = self.active_positions.get(symbol, {})
                    if position_info:
                        entry_price = position_info['entry_price']
                        pnl = (current_price - entry_price) * current_qty
                        pnl_percent = (pnl / (entry_price * current_qty)) * 100
                        position_age = current_time - position_info['entry_time']

                        logger.info(f"""Current Position Status:
- Quantity: {current_qty} shares
- Market Value: ${current_value:.2f}
- Entry Price: ${entry_price:.2f}
- Current Price: ${current_price:.2f}
- P&L: ${pnl:.2f} ({pnl_percent:.2f}%)
- Entry Time: {position_info['entry_time']}
- Position Age: {position_age}""")

                logger.info(f"Waiting {check_interval} seconds until next check...")
                time.sleep(check_interval)

            except KeyboardInterrupt:
                logger.info("Stopping bot...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                logger.error("Retrying in 5 minutes...")
                time.sleep(300)


if __name__ == "__main__":
    try:
        # Create and run the trading bot
        bot = TrendFollowingBot(trailing_stop_percent=0.05)  # 5% trailing stop
        bot.run_strategy("TSLA", check_interval=300)  # Check every 5 minutes
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")