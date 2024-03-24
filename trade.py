import ccxt
import sys
import time
import argparse
import logging
import config
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="logs/trading-%s.log" % time.strftime("%Y-%m-%d %H:%M:%S"),
    filemode="a",
)

# Set up logging to the console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

# Log basic information
logging.info("Python Version: %s", sys.version)
logging.info("CCXT Version: %s", ccxt.__version__)

# Log CCXT exchanges
logging.info("Loading script...")
logging.debug("Exchanges: %s", ccxt.exchanges)

# Log config
logging.debug("Config: %s", config.__dict__)

# Create an argument parser
parser = argparse.ArgumentParser(description="Weird Trading Bot")

# Add command-line arguments
parser.add_argument("--exchange", help="Exchange name", default="binance")
parser.add_argument("--pair", help="Trading pair")
parser.add_argument("--market_type", choices=["spot", "futures"], help="Market type", default="spot")
parser.add_argument("--side", choices=["LONG", "long", "SHORT", "short", "BUY", "buy", "SELL", "sell"], help="Trade side", default="BUY")
parser.add_argument("--entry_price", type=float, help="Entry price")
parser.add_argument("--stop_loss_price", type=float, help="Stop loss price")
parser.add_argument("--take_profit_price", type=float, help="Take profit price")
parser.add_argument("--order_size_percentage", type=float, help="Order size percentage", default=100)

# Parse the command-line arguments
args = parser.parse_args()

# Log command-line arguments
logging.info("Command-line arguments: %s", args)

# Access the parsed arguments
exchange_name: str = args.exchange.lower() if args.exchange else "binance"
pair: str = args.pair.upper()
market_type: str = args.market_type.lower() if args.market_type else "spot"
side: str = args.side.upper() if args.side else "BUY"
entry_price: float = args.entry_price
stop_loss_price: float = args.stop_loss_price
take_profit_price: float = args.take_profit_price
order_size_percentage: float = args.order_size_percentage

class WeirdAITradingBot:
    def __init__(self, exchange_name: str, pair: str, market_type: str, side: str, entry_price: float, stop_loss_price: float, take_profit_price: float, order_size_percentage: int):
        logging.info("Initializing CryptoTradingBot...")

        # Set the exchange name
        self.exchange_name: str = exchange_name or config.exchange_name or "binance"
        logging.info("Exchange Name: %s", self.exchange_name)
        # Set the trading pair
        self.pair: str = pair
        logging.info("Trading Pair: %s", self.pair)
        # Set the market type
        self.market_type: str = market_type
        logging.info("Market Type: %s", self.market_type)
        # Set the side
        self.side: str = side
        logging.info("Side: %s", self.side)
        # Set the opposite side
        self.opposite_side: str = "SELL" if self.side == "BUY" else "BUY"
        logging.debug("Opposite Side: %s", self.opposite_side)
        # Set the entry price
        self.entry_price: float = entry_price
        logging.info("Entry Price: %f", self.entry_price)
        # Set the stop loss
        self.stop_loss_price: float = stop_loss_price
        logging.info("Stop Loss: %f", self.stop_loss_price)
        # Set the take profit
        self.take_profit_price: float = take_profit_price
        logging.info("Take Profit: %f", self.take_profit_price)
        # Set the order size percentage
        if side == "BUY" and order_size_percentage == 100:
            order_size_percentage = 99
        logging.info("Order Size Percentage: %i%%", order_size_percentage)
        # Set the order size percentage decimal
        self.order_size_percentage: float = float(order_size_percentage / 100)
        logging.info("Order Size Percentage Decimal: %f", self.order_size_percentage)

        # Load the config
        self.load_trader_config()

        # Validate the config
        self.validate_trading_parms()

    def load_trader_config(self):
        # Load the config
        logging.info("Loading trader config...")

        # Set the maximum number of retries and polling interval
        self.max_retries: int = config.max_retries or 30 # 30 seconds * 10 = 5 minutes
        logging.info("Max Retries: %d", self.max_retries)
        # Set the polling interval
        self.polling_interval: int = config.polling_interval or 10 # 10 seconds
        logging.info("Polling Interval: %d", self.polling_interval)

    def validate_trading_parms(self):
        # Validate the config
        logging.info("Validating config...")

        # Validate the exchange name
        if self.exchange_name not in ccxt.exchanges:
            logging.error("Invalid exchange: %s", self.exchange_name)
            parser.error("Invalid exchange: " + self.exchange_name)

        # Validate the order size higher than 0 and lower or equal to 100
        if self.order_size_percentage <= 0 or self.order_size_percentage > 100:
            logging.error("Invalid order size percentage. Must be higher than 0 and lower or equal to 100.")
            parser.error("Invalid order size percentage. Must be higher than 0 and lower or equal to 100.")

        # Check if the user has configured the API key and secret
        if config.api_key == "" or config.api_secret == "":
            logging.error("Please configure your API key and secret in config.py")
            raise RuntimeError("Please configure your API key and secret in config.py")

        # Validate side
        if self.side in ["LONG", "SHORT"]:
            self.side = "BUY" if self.side == "LONG" else "SELL"

        if self.side not in ["BUY", "SELL"]:
            logging.error("Invalid side. Must be 'BUY' or 'SELL'.")
            parser.error("Invalid side. Must be 'BUY' or 'SELL'.")

        # Validate entry, stop loss, and take profit prices
        if self.entry_price <= 0 or self.stop_loss_price <= 0 or self.take_profit_price <= 0:
            logging.error("Invalid price. Prices must be greater than zero.")
            raise RuntimeError("Invalid price. Prices must be greater than zero.")

    def initialize_exchange(self):
        # Initialize the exchange
        logging.info("Initializing the exchange...")
        try:
            self.exchange_class = getattr(ccxt, self.exchange_name)
            self.exchange = self.exchange_class(
                {
                    "apiKey": config.api_key,
                    "secret": config.api_secret,
                    "enableRateLimit": True,
                    "options": {
                        "defaultType": "spot",
                        "adjustForTimeDifference": True
                    },
                }
            )
            logging.info("Exchange initialized successfully.")
        except ccxt.ExchangeError as e:
            logging.error("Failed to initialize the exchange: %s", str(e))
            raise RuntimeError("Failed to initialize the exchange: " + str(e))
        except ccxt.AuthenticationError as e:
            logging.error("Authentication failed: %s", str(e))
            raise RuntimeError("Authentication failed: " + str(e))
        except ccxt.BaseError as e:
            logging.error("Failed to initialize the exchange: %s", str(e))
            raise RuntimeError("Failed to initialize the exchange: " + str(e))

    def load_market_details(self):
        # Load market details
        logging.info("Loading market details...")
        try:
            self.exchange.load_markets()
            logging.info("Market details loaded successfully.")
        except ccxt.ExchangeError as e:
            logging.error("Failed to load market details: %s", str(e))
            raise RuntimeError("Failed to load market details: " + str(e))
        except ccxt.BaseError as e:
            logging.error("Failed to load market details: %s", str(e))
            raise RuntimeError("Failed to load market details: " + str(e))

    def load_trading_pair(self):
        # Validate the pair
        if self.pair not in self.exchange.symbols:
            logging.error(f"Invalid trading pair: {self.pair}")
            raise ValueError(f"Invalid trading pair: {self.pair}")

        # Get the market details
        logging.info("Loading pair market details...")
        try:
            self.market = self.exchange.market(self.pair)
            self.pair = self.market['symbol']
            self.pair_raw = self.market['id']
            logging.info("Pair market details: %s", self.market)
            logging.info("Pair market precision: %s", self.market["precision"])
            logging.info("Pair market limits: %s", self.market["limits"])
            logging.info("Pair market info: %s", self.market["info"])
        except ccxt.ExchangeError as e:
            logging.error("Failed to get pair market details: %s", str(e))
            raise RuntimeError("Failed to get pair market details: " + str(e))

        # Check if the trading pair is active
        if not self.market["active"]:
            logging.error(f"Inactive trading pair: {self.pair}")
            raise ValueError(f"Inactive trading pair: {self.pair}")

        # Validate that pair matches config market type spot
        if self.market_type == "spot" and not self.market["spot"]:
            logging.error(f"Invalid trading pair: {self.pair}")
            raise ValueError(f"Invalid trading pair: {self.pair}")

        # Validate that pair matches config market type futures
        if self.market_type == "futures" and not self.market["future"]:
            logging.error(f"Invalid trading pair: {self.pair}")
            raise ValueError(f"Invalid trading pair: {self.pair}")
        
        # Validate that pair matches config market type margin
        if self.market_type == "margin" and not self.market["margin"]:
            logging.error(f"Invalid trading pair: {self.pair}")
            raise ValueError(f"Invalid trading pair: {self.pair}")
        
        # Validate that pair matches config market type delivery
        if self.market_type == "delivery" and not self.market["delivery"]:
            logging.error(f"Invalid trading pair: {self.pair}")
            raise ValueError(f"Invalid trading pair: {self.pair}")
        
        # Set quantity and price precision
        self.quantity_precision = self.market["precision"]["amount"]
        logging.info("Quantity precision: %f", self.quantity_precision)
        self.price_precision = self.market["precision"]["price"]
        logging.info("Price precision: %f", self.price_precision)

        # Get the quote currency from the trading pair
        logging.info("Getting base and quote currencies...")
        self.base_currency, self.quote_currency = self.pair.split("/")
        logging.info("Base Currency: %s", self.base_currency)
        logging.info("Quote Currency: %s", self.quote_currency)

    def fetch_balance(self):
        # Fetch the account balance
        logging.info("Fetching account balance...")
        try:
            self.balance = self.exchange.fetch_balance()
            logging.debug("Account balance: %s", self.balance)
        except ccxt.ExchangeError as e:
            logging.error("Failed to fetch account balance: %s", str(e))
            raise RuntimeError("Failed to fetch account balance: " + str(e))
        
    def load_account_balances(self):
        self.fetch_balance()

        # Get the available quote balance
        logging.info("Getting the available quote balance...")
        self.available_quote_balance: float = self.balance["free"][self.quote_currency]
        logging.info("Available quote balance: %f %s", self.available_quote_balance, self.quote_currency)

        # Get the available base balance
        logging.info("Getting the available base balance...")
        self.available_base_balance: float = self.balance["free"][self.base_currency]
        logging.info("Available base balance: %f %s", self.available_base_balance, self.base_currency)

    def calculate_trade_quantity(self):
        # Calculate the quantity based on the entry price
        logging.info("Calculating the initial trade quantity...")
        if self.side == "BUY":
            # Use available quote balance for buying based on the order size percentage
            logging.info("Buying based on the order size percentage...")
            self.quantity: float = (self.available_quote_balance * self.order_size_percentage) / self.entry_price
            logging.info("Trade quantity: %f %s", self.quantity, self.base_currency)
        elif self.side == "SELL":
            # Use available base balance for selling based on the order size percentage
            logging.info("Selling based on the order size percentage...")
            self.quantity: float = self.available_base_balance
            logging.info("Trade quantity: %f %s", self.quantity, self.quote_currency)

    def validate_trade_quantity(self):
        # Validate and set the trade quantity based on the step size
        logging.info("Validating the trade quantity based on the step size...")
        try:
            if self.quantity_precision and self.quantity_precision > 0:
                self.quantity = float(self.exchange.amount_to_precision(
                    self.pair, self.quantity
                ))
                logging.info("Trade quantity: %f %s", self.quantity, self.base_currency)
        except ccxt.ExchangeError as e:
            logging.error(
                "Failed to validate the trade quantity based on the step size: %s", str(e)
            )
            raise RuntimeError(
                "Failed to validate the trade quantity based on the step size: " + str(e)
            )
        
        # Validate the trade quantity based on the available balance
        logging.info("Validating the trade quantity based on the available balance...")
        try:
            logging.info("Available quote balance: %f %s", self.available_quote_balance, self.quote_currency)
            if self.side == "BUY" and self.available_quote_balance < self.entry_price * self.quantity:
                logging.error("Insufficient funds: %f %s", self.available_quote_balance, self.quote_currency)
                raise ValueError("Insufficient funds: " + str(self.available_quote_balance) + " " + str(self.quote_currency))
            if self.side == "SELL" and self.available_base_balance < self.quantity:
                logging.error("Insufficient funds: %f %s", self.available_base_balance, self.base_currency)
                raise ValueError("Insufficient funds: " + str(self.available_base_balance) + " " + str(self.base_currency))
        except ccxt.ExchangeError as e:
            logging.error(
                "Failed to validate the trade quantity based on the available balance: %s",
                str(e),
            )
            raise RuntimeError(
                "Failed to validate the trade quantity based on the available balance: "
                + str(e)
            )

        # Validate the trade quantity based on the minimum notional value
        logging.info("Validating the trade quantity based on the minimum notional value...")
        try:
            self.min_notional: float = self.market["limits"]["cost"]["min"]
            logging.info("Minimum notional value: %f %s", self.min_notional, self.quote_currency)
            if self.side == "BUY" and self.entry_price * self.quantity < self.min_notional:
                logging.error("Quantity does not meet minimum notional value: %f %s", self.entry_price * self.quantity, self.quote_currency)
                raise ValueError("Invalid notional value: " + str(self.entry_price * self.quantity) + " " + str(self.quote_currency))
            if self.side == "SELL" and self.quantity < self.min_notional:
                logging.error("Quantity does not meet minimum notional value: %f %s", self.quantity, self.quote_currency)
                raise ValueError("Invalid notional value: " + str(self.quantity) + " " + str(self.quote_currency))
        except ccxt.ExchangeError as e:
            logging.error(
                "Failed to validate the trade quantity based on the minimum notional value: %s",
                str(e),
            )
            raise RuntimeError(
                "Failed to validate the trade quantity based on the minimum notional value: "
                + str(e)
            )

        # Validate the trade quantity based on the minimum quantity
        logging.info("Validating the trade quantity based on the minimum quantity...")
        try:
            self.min_quantity: float = self.market["limits"]["amount"]["min"]
            logging.info("Minimum quantity: %f", self.min_quantity)
            if self.quantity < self.min_quantity:
                logging.error("Invalid quantity: %f", self.quantity)
                raise ValueError("Invalid quantity: " + str(self.quantity))
        except ccxt.ExchangeError as e:
            logging.error(
                "Failed to validate the trade quantity based on the minimum quantity: %s",
                str(e),
            )
            raise RuntimeError(
                "Failed to validate the trade quantity based on the minimum quantity: "
                + str(e)
            )

        # Validate the trade quantity based on the maximum quantity
        logging.info("Validating the trade quantity based on the maximum quantity...")
        try:
            self.max_quantity: float = self.market["limits"]["amount"]["max"]
            logging.info("Maximum quantity: %f", self.max_quantity)
            if self.quantity > self.max_quantity:
                logging.error("Invalid quantity: %f", self.quantity)
                raise ValueError("Invalid quantity: " + str(self.quantity))
        except ccxt.ExchangeError as e:
            logging.error(
                "Failed to validate the trade quantity based on the maximum quantity: %s",
                str(e),
            )
            raise RuntimeError(
                "Failed to validate the trade quantity based on the maximum quantity: "
                + str(e)
            )

    def validate_entry_price(self):
        # Validate the entry price based on the tick size
        logging.info("Validating the entry price based on the tick size...")
        try:
            self.entry_price_precision: float = self.market["precision"]["price"]
            logging.info("Entry price precision: %f", self.entry_price_precision)
            if self.entry_price_precision and self.entry_price_precision > 0:
                entry_price = float(self.exchange.price_to_precision(
                    self.pair, self.entry_price
                ))
                logging.info("Entry price: %f", entry_price)
        except ccxt.ExchangeError as e:
            logging.error(
                "Failed to validate the entry price based on the tick size: %s", str(e)
            )
            raise RuntimeError(
                "Failed to validate the entry price based on the tick size: " + str(e)
            )

        # Validate the entry price based on the minimum price
        logging.info("Validating the entry price based on the minimum price...")
        try:
            self.min_entry_price: float = self.market["limits"]["price"]["min"]
            logging.info("Minimum entry price: %f", self.min_entry_price)
            if self.entry_price < self.min_entry_price:
                logging.error("Invalid entry price: %f", self.entry_price)
                raise ValueError("Invalid entry price: " + str(self.entry_price))
        except ccxt.ExchangeError as e:
            logging.error(
                "Failed to validate the entry price based on the minimum price: %s", str(e)
            )
            raise RuntimeError(
                "Failed to validate the entry price based on the minimum price: " + str(e)
            )

        # Validate the entry price based on the maximum price
        logging.info("Validating the entry price based on the maximum price...")
        try:
            self.max_entry_price: float = self.market["limits"]["price"]["max"]
            logging.info("Maximum price: %f", self.max_entry_price)
            if self.entry_price > self.max_entry_price:
                logging.error("Invalid entry price: %f", self.entry_price)
                raise ValueError("Invalid entry price: " + str(self.entry_price))
        except ccxt.ExchangeError as e:
            logging.error(
                "Failed to validate the entry price based on the maximum price: %s", str(e)
            )
            raise RuntimeError(
                "Failed to validate the entry price based on the maximum price: " + str(e)
            )

    def validate_stop_loss_and_take_profit_prices(self):
        # Validate the stop loss and take profit prices based on the tick size
        logging.info("Validating the stop loss and take profit prices based on the tick size...")
        try:
            if self.price_precision and self.price_precision > 0:
                self.stop_loss_price = float(self.exchange.price_to_precision(
                    self.pair, self.stop_loss_price
                ))
                self.take_profit_price = float(self.exchange.price_to_precision(
                    self.pair, self.take_profit_price
                ))
        except ccxt.ExchangeError as e:
            logging.error(
                "Failed to validate the stop loss and take profit prices based on the tick size: %s",
                str(e),
            )
            raise RuntimeError(
                "Failed to validate the stop loss and take profit prices based on the tick size: "
                + str(e)
            )

        # Validate the stop loss and take profit prices based on the entry price
        logging.info("Validating the stop loss and take profit prices based on the entry price...")
        try:
            if self.stop_loss_price >= self.entry_price:
                logging.error("Invalid stop loss price: %f", self.stop_loss_price)
                raise ValueError("Invalid stop loss price: " + str(self.stop_loss_price))
            if self.take_profit_price <= self.entry_price:
                logging.error("Invalid take profit price: %f", self.take_profit_price)
                raise ValueError("Invalid take profit price: " + str(self.take_profit_price))
            logging.info("Stop loss and take profit prices validated successfully.")
        except ccxt.ExchangeError as e:
            logging.error(
                "Failed to validate the stop loss and take profit prices: %s", str(e)
            )
            raise RuntimeError(
                "Failed to validate the stop loss and take profit prices: " + str(e)
            )

        # Validate and set the stop loss and take profit prices based on the minimum price precision
        logging.info(
            "Validating the stop loss and take profit prices based on the minimum price precision..."
        )
        try:
            logging.info("Price precision: %f", self.price_precision)
            if self.price_precision and self.price_precision > 0:
                self.stop_loss_price = float(self.exchange.price_to_precision(
                    self.pair, self.stop_loss_price
                ))
                self.take_profit_price = float(self.exchange.price_to_precision(
                    self.pair, self.take_profit_price
                ))
        except ccxt.ExchangeError as e:
            logging.error(
                "Failed to validate the stop loss and take profit prices based on the minimum price precision: %s",
                str(e),
            )
            raise RuntimeError(
                "Failed to validate the stop loss and take profit prices based on the minimum price precision: "
                + str(e)
            )

        # Validate the stop loss and take profit prices based on the minimum notional value
        logging.info(
            "Validating the stop loss and take profit prices based on the minimum notional value..."
        )
        try:
            self.min_notional = self.market["limits"]["cost"]["min"]
            logging.info("Minimum notional value: %f %s", self.min_notional, self.quote_currency)
            if self.stop_loss_price * self.quantity < self.min_notional:
                logging.error(
                    "Invalid stop loss price. Minimum notional value: %f", self.min_notional
                )
                raise ValueError(
                    "Invalid stop loss price. Minimum notional value: " + str(self.min_notional)
                )
            if self.take_profit_price * self.quantity < self.min_notional:
                logging.error(
                    "Invalid take profit price. Minimum notional value: %f", self.min_notional
                )
                raise ValueError(
                    "Invalid take profit price. Minimum notional value: " + str(self.min_notional)
                )
        except ccxt.ExchangeError as e:
            logging.error(
                "Failed to validate the stop loss and take profit prices based on the minimum notional value: %s",
                str(e),
            )
            raise RuntimeError(
                "Failed to validate the stop loss and take profit prices based on the minimum notional value: "
                + str(e)
            )

        # Validate the stop loss and take profit prices based on the minimum price
        logging.info("Validating the stop loss and take profit prices based on the minimum price...")
        try:
            self.min_price = self.market["limits"]["price"]["min"]
            logging.info("Minimum price: %f", self.min_price)
            if self.stop_loss_price < self.min_price:
                logging.error("Invalid stop loss price: %f", self.stop_loss_price)
                raise ValueError("Invalid stop loss price: " + str(self.stop_loss_price))
            if self.take_profit_price < self.min_price:
                logging.error("Invalid take profit price: %f", self.take_profit_price)
                raise ValueError("Invalid take profit price: " + str(self.take_profit_price))
        except ccxt.ExchangeError as e:
            logging.error(
                "Failed to validate the stop loss and take profit prices based on the minimum price: %s",
                str(e),
            )
            raise RuntimeError(
                "Failed to validate the stop loss and take profit prices based on the minimum price: "
                + str(e)
            )

        # Validate the stop loss and take profit prices based on the maximum price
        logging.info("Validating the stop loss and take profit prices based on the maximum price...")
        try:
            self.max_price = self.market["limits"]["price"]["max"]
            logging.info("Maximum price: %f", self.max_price)
            if self.stop_loss_price > self.max_price:
                logging.error("Invalid stop loss price: %f", self.stop_loss_price)
                raise ValueError("Invalid stop loss price: " + str(self.stop_loss_price))
            if self.take_profit_price > self.max_price:
                logging.error("Invalid take profit price: %f", self.take_profit_price)
                raise ValueError("Invalid take profit price: " + str(self.take_profit_price))
        except ccxt.ExchangeError as e:
            logging.error(
                "Failed to validate the stop loss and take profit prices based on the maximum price: %s",
                str(e),
            )
            raise RuntimeError(
                "Failed to validate the stop loss and take profit prices based on the maximum price: "
                + str(e)
            )

    def place_entry_order(self):
        # Place the entry order
        logging.info("place_entry_order()")

        # Set the entry order params
        self.entry_order_params = {
            "symbol": self.pair,
            "type": "limit",
            "side": self.side,
            "amount": self.quantity,
            "price": self.entry_price,
            "params": {
                "timeInForce": "GTC",
                'triggerPrice': self.entry_price
            },
        }
        logging.info("Entry order params: %s", self.entry_order_params)
        
        # Place the entry order
        logging.info("Placing the entry order...")
        self.entry_order = self.create_order(self.entry_order_params)
        logging.info("Entry order placed: %s", self.entry_order)

        # Polling loop to monitor the entry order
        logging.info("Monitoring the entry order...")
        self.retries = 0
        while self.retries < self.max_retries:
            self.retries += 1

            # Announce iteration and number of seconds elapsed
            logging.info(
                "Entry Iteration %d/%d. Waiting %d seconds...",
                self.retries,
                self.max_retries,
                self.polling_interval,
            )
            # Polling interval in seconds
            time.sleep(self.polling_interval)

            # Fetch the entry order status
            logging.info("Fetching entry order status...")
            self.entry_order_status = self.fetch_order(self.entry_order["id"], self.entry_order["symbol"])
            logging.info("Entry order status: %s", self.entry_order_status)

            # Check if the entry order is filled
            if self.entry_order_status["status"] == "closed":
                logging.info("Entry order executed successfully.")
                break
            if self.entry_order_status["status"] == "canceled":
                logging.error("Entry order canceled outside the script.")
                sys.exit(1)

        # Handle the case when the maximum number of retries is reached
        if self.retries == self.max_retries:
            logging.error(
                "Entry Timeout: Failed to execute entry order for %s within %d retries.",
                self.pair,
                self.max_retries,
            )
            logging.error("Cancelling entry order...")
            self.entry_order_cancel = self.cancel_order(self.entry_order_status['id'], self.entry_order_status['symbol'])
            logging.error("Entry order canceled: %s", self.entry_order_cancel)
            sys.exit(1)

    def place_stop_loss_and_take_profit_orders(self):
        # Place the stop loss and take profit orders
        logging.info("place_stop_loss_and_take_profit_orders()")

        # Set the stop loss order params
        self.oco_order_params = {
            "symbol": self.pair_raw,
            "side": self.opposite_side,
            "quantity": self.exchange.amount_to_precision(self.pair, self.quantity),
            "price": self.exchange.price_to_precision(self.pair, self.take_profit_price),
            "stopPrice": self.exchange.price_to_precision(self.pair, self.stop_loss_price),
            "stopLimitPrice": self.exchange.price_to_precision(self.pair, self.stop_loss_price),
            "stopLimitTimeInForce": "GTC",
        }
        logging.info("OCO order params: %s", self.oco_order_params)

        self.retries = 0
        while self.retries < self.max_retries * 100:
            self.retries += 1

            # Announce iteration and number of seconds elapsed
            logging.info(
                "OCO Iteration %d/%d. Waiting %d seconds...",
                self.retries,
                self.max_retries,
                self.polling_interval,
            )

            try:
                # Place the entry order
                logging.info("Placing the OCO order...")
                self.oco_order = self.create_oco_order(self.oco_order_params)
                logging.info("OCO order placed: %s", self.oco_order)
                break
            except Exception as e:
                # if error has "Account has insufficient balance for requested action.":
                #   reduce quantity by min_quantity and try again
                if "Account has insufficient balance for requested action." in str(e):
                    pass
                    logging.error("Failed to place OCO order: %s", str(e))
                    logging.error("Reducing quantity by %f %s", self.min_quantity, self.base_currency)
                    self.quantity -= self.min_quantity * self.retries
                    self.oco_order_params["quantity"] = self.exchange.amount_to_precision(self.pair, self.quantity)
                    logging.error("Reduced quantity: %f %s", self.oco_order_params["quantity"], self.base_currency)
    
                # Announce iteration and number of seconds elapsed
                logging.info("Waiting %d seconds...", self.polling_interval)
                # Polling interval in seconds
                time.sleep(self.polling_interval)

        # Handle the case when the oco_order isn't successful
        if self.oco_order is None:
            logging.error(
                "OCO Timeout: Failed to execute OCO order (id: %s) for %s within %d retries.",
                self.oco_order["id"],
                self.oco_order["symbol"],
                self.max_retries,
            )
            sys.exit(1)

        # Set the stop loss and take profit order ids
        self.stop_loss_order = self.oco_order["orderReports"][0]
        self.stop_loss_order['id'] = self.stop_loss_order['orderId']
        self.take_profit_order = self.oco_order["orderReports"][1]
        self.take_profit_order['id'] = self.take_profit_order['orderId']

        # Polling loop to monitor the stop loss and take profit orders
        logging.info("Monitoring the stop loss and take profit orders...")
        while True:
            # Announce iteration and number of seconds elapsed
            logging.info("Waiting %d seconds...", self.polling_interval)
            # Polling interval in seconds
            time.sleep(self.polling_interval)  # Polling interval in seconds

            # Fetch the latest order status
            logging.info("Fetching stop loss order status...")
            try:
                self.stop_loss_order_status = self.fetch_order(self.stop_loss_order["id"], self.stop_loss_order["symbol"])
                logging.info("Stop loss order status: %s", self.stop_loss_order_status)

                logging.info("Fetching take profit order status...")
                self.take_profit_order_status = self.fetch_order(self.take_profit_order["id"], self.take_profit_order["symbol"])
                logging.info("Take profit order status: %s",self.take_profit_order_status)
            except Exception as e:
                pass

            # Check if the stop loss or take profit order is filled
            if self.stop_loss_order_status["status"] == "closed":
                logging.info("Stop loss order executed. Trade completed.")
                break
            elif self.take_profit_order_status["status"] == "closed":
                logging.info("Take profit order executed. Trade completed.")
                break

    def create_oco_order(self, params):
        # Place the OCO order
        logging.info("Placing stop loss and take profit orders (OCO)...")
        try:
            logging.info("Stop loss and take profit orders (OCO) parameters: %s", params)
            oco_order = self.exchange.privatePostOrderOco(params)
            logging.info("Stop loss and take profit orders (OCO) placed: %s", oco_order)
            return oco_order
        except ccxt.ExchangeError as e:
            logging.error("Failed to place stop loss or take profit orders: %s", str(e))
            raise RuntimeError("Failed to place stop loss or take profit orders: " + str(e))
        except ccxt.InvalidOrder as e:
            logging.errror("Invalid order: %s", str(e))
            raise ValueError("Invalid order: " + str(e))
        except ccxt.InsufficientFunds as e:
            logging.error("Insufficient funds: %s", str(e))
            raise ValueError("Insufficient funds: " + str(e))
        except ccxt.NetworkError as e:
            logging.error("Network error occurred: %s", str(e))
            raise RuntimeError("Network error occurred: " + str(e))
        except ccxt.BaseError as e:
            logging.error("Failed to place entry order: %s", str(e))
            raise RuntimeError("Failed to place entry order: " + str(e))
    
    def create_order(self, params):
         # Place the order
        logging.info("Placing order...")
        try:
            logging.info("Order parameters: %s", params)
            order = self.exchange.create_order(**params)
            logging.info("Order placed: %s", order)
            return order
        except ccxt.ExchangeError as e:
            logging.error("Failed to place order: %s", str(e))
            raise RuntimeError("Failed to place order: " + str(e))
        except ccxt.InvalidOrder as e:
            logging.error("Invalid order: %s", str(e))
            raise ValueError("Invalid order: " + str(e))
        except ccxt.InsufficientFunds as e:
            logging.error("Insufficient funds: %s", str(e))
            raise ValueError("Insufficient funds: " + str(e))
        except ccxt.NetworkError as e:
            logging.error("Network error occurred: %s", str(e))
            raise RuntimeError("Network error occurred: " + str(e))
        except ccxt.BaseError as e:
            logging.error("Failed to place entry order: %s", str(e))
            raise RuntimeError("Failed to place entry order: " + str(e))
        
    def fetch_order(self, order_id, symbol):
        # Fetch order status
        logging.info("Fetching order status...")
        try:
            entry_order_status = self.exchange.fetch_order(
                order_id, symbol
            )
            logging.info("Order status: %s", entry_order_status)
            return entry_order_status
        except ccxt.ExchangeError as e:
            logging.error("Failed to fetch order status: %s", str(e))
            raise RuntimeError("Failed to fetch order status: " + str(e))
        
    def cancel_order(self, order_id, symbol):
        # Cancel order
        logging.info("Canceling order...")
        try:
            cancel_order_status = self.exchange.cancel_order(order_id, symbol)
            logging.info("Entry order canceled: %s", cancel_order_status)
            return cancel_order_status
        except ccxt.ExchangeError as e:
            logging.error("Failed to cancel entry order: %s", str(e))
            raise RuntimeError("Failed to cancel entry order: " + str(e))

    def cancel_remaining_orders(self):
        # Cancel any remaining orders
        logging.info("Canceling remaining orders...")
        try:
            logging.info("Canceling remaining stop loss order if it exists...")
            self.cancel_order(self.stop_loss_order["id"], self.stop_loss_order["symbol"])
            logging.info("Canceling remaining stop loss order if it exists...")
            self.cancel_order(self.take_profit_order["id"], self.take_profit_order["symbol"])
            logging.info("Remaining orders canceled.")
        except ccxt.ExchangeError as e:
            logging.error("Failed to cancel remaining orders: %s", str(e))
            raise RuntimeError("Failed to cancel remaining orders: " + str(e))

    def main(self):
        # Validate the exchange
        self.initialize_exchange()

        # Load market details
        self.load_market_details()

        # Validate the trading pair
        self.load_trading_pair()

        # Fetch the account balance
        self.load_account_balances()

        # Calculate the trade quantity
        self.calculate_trade_quantity()

        # Validate the trade quantity
        self.validate_trade_quantity()

        # Validate the entry price
        self.validate_entry_price()

        # Validate the stop loss and take profit prices
        self.validate_stop_loss_and_take_profit_prices()

        # Place the entry order
        self.place_entry_order()

        # Place the stop loss and take profit orders
        self.place_stop_loss_and_take_profit_orders()

        # Cancel any remaining orders
        self.cancel_remaining_orders()

if __name__ == "__main__":
    # Run the trading bot
    try:
        bot = WeirdAITradingBot(
            exchange_name,
            pair,
            market_type,
            side,
            entry_price,
            stop_loss_price,
            take_profit_price,
            order_size_percentage,
        )
        bot.main()
    except Exception as e:
        logging.error("Failed to complete the trade: %s", str(e))
        raise RuntimeError("Failed to complete the trade: " + str(e))
    
    logging.info("Trade completed successfully.")
    sys.exit(1)