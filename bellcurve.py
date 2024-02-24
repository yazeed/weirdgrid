import json
import os
from datetime import datetime
import ccxt
import numpy as np

def log_message(exchange, symbol, message):
    """
    Log a message to a file.
    """
    # Adjusted to include exchange name in log directory path
    log_dir = f"logs/{exchange.name}"
    os.makedirs(log_dir, exist_ok=True)
    
    date_str = datetime.now().strftime("%Y-%m-%d")
    folder_path = f"{log_dir}/{date_str}"
    os.makedirs(folder_path, exist_ok=True)
    
    stripped_symbol = symbol.replace("/", "-")
    filename = f"{folder_path}/{stripped_symbol}.log"
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filename, "a") as log_file:
        log_file.write(f"[{timestamp}] {message}\n")

def load_config(config_path='config.json'):
    """
    Load the API configuration from a JSON file.
    """
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def initialize_exchange(exchange_name, config):
    """
    Initialize the exchange with API keys from the configuration.
    """
    # Validate that config contains key with exchange_name
    if exchange_name not in config:
        print(f"Error: apiKey and secret configuration for {exchange_name} not found.")
        exit()
    
    # Initialize the exchange
    exchange_config = {
        'apiKey': config[exchange_name]['apiKey'],
        'secret': config[exchange_name]['secretKey'],
        'enableRateLimit': True,
    }

    if 'password' in config[exchange_name]:
        exchange_config['password'] = config[exchange_name]['password']

    exchange = getattr(ccxt, exchange_name)(exchange_config)

    # Validate that the exchange is working
    if not exchange.check_required_credentials():
        print(f"Error: Exchange {exchange_name} is not working.")
        exit()

    return exchange

def validate_symbol(exchange, symbol):
    """
    Validate the trading symbol and fetch market details.
    """
    exchange.load_markets()
    if symbol in exchange.markets:
        return exchange.markets[symbol]
    else:
        print(f"Symbol {symbol} not found on {exchange.id}.")
        return None

def fetch_and_display_balances(exchange, symbol):
    """
    Fetch the current balance of the base and quote assets and display them.
    """
    # Fetch the market details for the symbol
    base, quote = symbol.split('/')
    # Fetch the current balance of the base and quote assets
    balance = exchange.fetch_balance()
    base_balance = balance.get(base, {}).get('free', 0)
    quote_balance = balance.get(quote, {}).get('free', 0)
    print(f"{base} Quantity available to allocate for this bell curve grid: {base_balance} {base}")
    print(f"{quote} Quantity available to allocate for this bell curve grid: {quote_balance} {quote}\n")
    return base, quote, base_balance, quote_balance

def fetch_current_price(exchange, symbol):
    """
    Fetch the current price of the symbol.
    """
    ticker = exchange.fetch_ticker(symbol)
    print(f"\nCurrent price of {symbol}: {ticker['last']}\n")
    return ticker['last']  # Using the last price as the current price

def decide_quantities(base, quote, base_balance, quote_balance, base_quantity, quote_quantity, current_price):
    """
    Determine the quantities to allocate for buy and sell orders.
    """
    if low_price < current_price < high_price:
        # Range straddles the current price; ask for both allocations
        base_quantity_input = input(f"Enter how much of your {base_balance} {base} to allocate for sell orders [{base_quantity}]: ")
        base_quantity = float(base_quantity_input) if base_quantity_input else base_quantity
        quote_quantity_input = input(f"Enter how much of your {quote_balance} {quote} to allocate for buy orders [{quote_quantity}]: ")
        quote_quantity = float(quote_quantity_input) if quote_quantity_input else quote_quantity
    elif high_price <= current_price:
        # All orders are buy orders; only QUOTE allocation needed
        quote_quantity_input = input(f"Enter how much of your {quote_balance} {quote} to allocate for buy orders [{quote_quantity}]: ")
        quote_quantity = float(quote_quantity_input) if quote_quantity_input else quote_quantity
        base_quantity = 0
    else:  # low_price >= current_price
        # All orders are sell orders; only BASE allocation needed
        base_quantity_input = input(f"Enter how much of your {base_balance} {base} to allocate for sell orders [{base_quantity}]: ")
        base_quantity = float(base_quantity_input) if base_quantity_input else base_quantity
        quote_quantity = 0

    return base_quantity, quote_quantity

def decide_peak_position(peak_position, total_orders, std_dev_dividor):
    """
    Decide on the peak position.
    """
    try:
        peak_position_input = input(f"Enter peak position (index or 'middle') [{peak_position}]: ")
        if peak_position_input.isdigit():
            peak_position = int(peak_position_input) if peak_position_input else peak_position

            # Validate peak_position
            if not (0 <= peak_position <= total_orders):
                print(f"Peak position {peak_position} out of range for {total_orders} orders.")
                decide_peak_position(peak_position, total_orders, std_dev_dividor)
        else:
            # if peak_position_input empty use peak_position, else peak_position is set to middle
            if peak_position_input:
                peak_position = 'middle'
    except ValueError:
        peak_position = 'middle'  # Default to middle on any conversion error

    std_dev_dividor_input = input(f"Enter the standard deviation of the distribution of orders [{std_dev_dividor}]: ")
    std_dev_dividor = float(std_dev_dividor_input) if std_dev_dividor_input else std_dev_dividor

    return peak_position, std_dev_dividor

def decide_prices_and_orders(low_price, high_price, total_orders):
    """
    Decide on the prices and orders.
    """
    # Adjusted user input for symbol, price range, total orders, and peak position
    low_price = float(input(f"Enter the low price [{low_price if low_price is not None else 'None'}]: ") or low_price)
    high_price = float(input(f"Enter the high price [{high_price if high_price is not None else 'None'}]: ") or high_price)
    total_orders = int(input(f"Enter the total number of orders [{total_orders}]: ") or total_orders)

    # Convert input values to appropriate types
    low_price = float(low_price)
    high_price = float(high_price)
    total_orders = int(total_orders)

    return low_price, high_price, total_orders

def create_bell_curve_orders(current_price, low_price, high_price, peak_position, total_orders, base_quantity, quote_quantity, std_dev_dividor, market_details, exchange, symbol):
    """
    Create the bell curve orders.
    """
    # Calculate the price points
    price_points = np.linspace(low_price, high_price, total_orders)
    avg_price = np.mean([low_price, high_price])

    # Adjust peak_position handling
    mean_index = total_orders // 2 if peak_position == 'middle' else int(peak_position)

    # Calculate distribution
    x_values = np.arange(total_orders)
    std_dev = total_orders / std_dev_dividor  # Adjusted for broader distribution
    distribution = np.exp(-(x_values - mean_index) ** 2 / (2 * std_dev ** 2))
    distribution /= distribution.sum()  # Normalize distribution

    orders = []
    min_qty = market_details['limits']['amount']['min']

    for i, price in enumerate(price_points):
        # Determine order type based on price relative to current market price
        order_type = "Buy" if price < current_price else "Sell"
        # Calculate quantity based on distribution and order type
        quantity = (quote_quantity / avg_price) * distribution[i] if order_type == "Buy" else (base_quantity * distribution[i])

        # Check if quantity meets min_qty requirement
        if quantity < min_qty:
            continue

        formatted_price = exchange.price_to_precision(symbol, price)
        formatted_quantity = exchange.amount_to_precision(symbol, quantity)

        formatted_price_float = float(formatted_price)
        formatted_quantity_float = float(formatted_quantity)

        stake_amount = float(formatted_quantity_float * formatted_price_float)

        if stake_amount < min_qty:
            continue

        orders.append((formatted_price_float, formatted_quantity_float, stake_amount, order_type))

    return orders

def print_orders_plan(exchange, symbol, orders):
    """
    Print the orders plan.
    """
    # Fetch the market details for the symbol
    base, quote = symbol.split('/')
    # Temporary storage for order plan messages
    order_plan_messages = []
    # Print orders plan
    print(f"\nBell Curve Grid - {exchange.name} - {symbol} Orders Plan:")
    print('-----------------------------------')
    for i, (price, quantity, stake_amount, order_type) in enumerate(orders, start=1):
        message = f"{i}. {order_type} Order - Price: {price} - Quantity: {quantity} {base} - Cost: {stake_amount} {quote}"
        print(message)
        order_plan_messages.append(message)
    
    # Log inputs
    log_inputs = '\n'.join([
        f'-----------------------------------',
        f'Number of Orders in Bell Curve Grid: {total_orders}',
        f'Peak Position: {peak_position}',
        f'Base Quantity: {base_quantity} {base}',
        f'Quote Quantity: {quote_quantity} {quote}',
        f'-----------------------------------',
        f'High Price: {high_price}',
        f'Current Price: {current_price}',
        f'Low Price: {low_price}',
        f'-----------------------------------'
    ])
    print(log_inputs)

    # Ask if the user is satisfied with the orders plan
    user_confirmation = input("\nAre you satisfied with the orders plan? (yes/no): ").lower()
    if user_confirmation == 'yes':
        log_message(exchange, symbol, f"Bell Curve Grid - {exchange.name} - {symbol} Orders Plan:")
        # If approved, log all order plan messages
        for message in order_plan_messages:
            log_message(exchange, symbol, message)

        log_message(exchange, symbol, log_inputs)

    return user_confirmation

def place_orders(exchange, symbol, orders, market_details):
    """
    Place buy and sell limit orders after user confirmation.
    """
    confirmation = input("Do you want to place these orders? (yes/no): ").lower()
    if confirmation.lower() != 'yes':
        print("Orders not placed.")
        return
    
    # Fetch the market details for the symbol
    base, quote = symbol.split('/')
    
    for i, (price, quantity, stake_amount, order_type) in enumerate(orders, start=1):
        # Apply precision
        formatted_price = exchange.price_to_precision(symbol, price)
        formatted_quantity = exchange.amount_to_precision(symbol, quantity)
        formatted_cost = exchange.cost_to_precision(symbol, stake_amount)
        
        # Convert to float for comparison
        price = float(formatted_price)
        quantity = float(formatted_quantity)
        cost = float(formatted_cost)

        # Print market details
        print(f"\nOrder Details for {symbol}:")
        print(f"-----------------------------------")
        print(f"Price: {price}")
        print(f"Quantity: {quantity} {base}")
        print(f"Cost: {cost} {quote}")
        print(f"-----------------------------------\n")
        
        # Validate against limits
        if not (market_details['limits']['amount']['min'] <= quantity <= market_details['limits']['amount']['max']):
            print(f"Quantity {quantity} out of range for {symbol}.")
            continue
        if (not (market_details['limits']['price']['min'] is None) or not (market_details['limits']['price']['max'] is None)) and (not (market_details['limits']['price']['min'] <= price <= market_details['limits']['price']['max'])):
            print(f"Price {price} out of range for {symbol}.")
            continue
        if not (market_details['limits']['cost']['min'] <= cost <= market_details['limits']['cost']['max']):
            print(f"Cost {cost} out of range for {symbol}.")
            continue
        
        # Place order
        try:
            if order_type == 'Buy':
                # Print buy order details
                print(f"{i}. Placing {order_type} Order - {symbol} - Price: {price} - Quantity: {quantity} {base} - Cost: {cost} {quote}")
                # Place buy order
                order = exchange.create_limit_buy_order(symbol, formatted_quantity, formatted_price)
            else:
                # Print sell order details
                print(f"{i}. Placing {order_type} Order - {symbol} - Price: {price} - Quantity: {quantity} {base} - Cost: {cost} {quote}")
                # Place sell order
                order = exchange.create_limit_sell_order(symbol, formatted_quantity, formatted_price)

            success_message = f"{i}. Order Placed: {order_type.capitalize()} {formatted_quantity} of {symbol} @ {formatted_price} for a total cost of {cost}."
            print(success_message)
            print(f"{i}. Order Details: ", order)
            log_message(exchange, symbol, success_message)
            log_message(exchange, symbol, f'{i}. Order Details: {order}')
        except Exception as e:
            failure_message = f"{i}. Failed to place {order_type.capitalize()} order for {symbol} at price {formatted_price} and quantity {formatted_quantity}.\n"
            print(failure_message)
            print(f"{i}. Error:", e)
            log_message(exchange, symbol, failure_message)
            log_message(exchange, symbol, f'{i}. Error: {e}')

# Load API configuration
config = load_config()

# Load default variables
default_exchange = config['defaults']['exchange']
low_price = None
low_price_percent = float(config['defaults']['lowPricePct'])
high_price = None
high_price_percent = float(config['defaults']['highPricePct'])
current_price = None
total_orders = int(config['defaults']['totalOrders'])
peak_position = config['defaults']['peakPosition']
base_quantity = 0
quote_quantity = 0
std_dev_dividor = float(config['defaults']['stdDevDividor'])

# User input for symbol
exchange_input = input(f"Enter the exchange name [{default_exchange}]: ").lower()
exchange_name = exchange_input if exchange_input else default_exchange

# Initialize CCXT exchange client with your API keys
exchange = initialize_exchange(exchange_name, config)

# User input for symbol
symbol = input("Enter the symbol (e.g., 'JUP/USDT'): ").upper()

# Validate the symbol and fetch market details
market_details = validate_symbol(exchange, symbol)
if not market_details:
    # Handle invalid symbol case here (e.g., request new input or exit)
    exit("Exiting due to invalid symbol.")

# Iterate until user inputs are valid
satisfied = False
while not satisfied:
    # Fetch current price and generate orders
    current_price = fetch_current_price(exchange, symbol)

    if current_price is None:
        print("Could not fetch current price for symbol. Please try again.")
        continue

    # If low_price and high_price are None, set low_price to 15% below current_price and high_price to 2.5% below current_price, rounded to same number of decimals on current_price
    if low_price is None and high_price is None:
        low_price = round(current_price * low_price_percent, len(str(current_price).split('.')[1]))
        high_price = round(current_price * high_price_percent, len(str(current_price).split('.')[1]))

    # Assume base_balance and quote_balance fetched from the exchange
    base, quote, base_balance, quote_balance = fetch_and_display_balances(exchange, symbol)

    # Adjusted user input for symbol, price range, total orders, and peak position
    low_price, high_price, total_orders = decide_prices_and_orders(low_price, high_price, total_orders)

    # Decide quantities to allocate to buy and sell orders
    base_quantity, quote_quantity = decide_quantities(base, quote, base_balance, quote_balance, base_quantity, quote_quantity, current_price)

    # Decide peak position on bell curve
    peak_position, std_dev_dividor = decide_peak_position(peak_position, total_orders, std_dev_dividor)

    # Assuming market_details is obtained and validated earlier in the script
    orders = create_bell_curve_orders(current_price, low_price, high_price, peak_position, total_orders, base_quantity, quote_quantity, std_dev_dividor, market_details, exchange, symbol)

    # Validate bell orders more than 1 order
    if len(orders) < 2 or len(orders) > 100:
        print("The bell curve grid requires at least 2 orders, and a maximum of 100 orders.")
        continue

    # Print orders plan
    user_confirmation = print_orders_plan(exchange, symbol, orders)

    satisfied = user_confirmation == 'yes'

# Place orders after confirmation
place_orders(exchange, symbol, orders, market_details)

# Exit the script
log_message(exchange, symbol, 'Done!')
print("Done!")
