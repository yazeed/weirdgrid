import json
import os
from datetime import datetime
import ccxt
import numpy as np
import time


def log_message(exchange, symbol, message):
    """
    Log a message to a file.
    """
    # Adjusted to include exchange name in log directory path
    log_dir = f"logs/{exchange}"
    os.makedirs(log_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    folder_path = f"{log_dir}/{date_str}"
    os.makedirs(folder_path, exist_ok=True)
    stripped_symbol = symbol.replace("/", "-")
    filename = f"{folder_path}/{stripped_symbol}.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, "a") as log_file:
        log_file.write(f"[{timestamp}] {message}\n")


def load_config(config_path="config.json"):
    """
    Load the API configuration from a JSON file.
    """
    with open(config_path, "r") as file:
        config = json.load(file)
    return config


def initialize_exchange(exchange_name, config):
    """
    Initialize the exchange object with the given name and configuration.
    """
    if exchange_name not in config:
        print(
            f"Error: apiKey and secretKey configuration for {exchange_name} not found."
        )
        exit()
    exchange_config = {
        "apiKey": config[exchange_name]["apiKey"],
        "secret": config[exchange_name]["secretKey"],
        "enableRateLimit": True,
        "options": {
            "defaultType": "spot",
        },
    }
    if "password" in config[exchange_name]:
        exchange_config["password"] = config[exchange_name]["password"]
    exchange = getattr(ccxt, exchange_name)(exchange_config)

    # Print all exchange object attributes and values
    if DEBUG_MODE:
        print(f"Exchange: {exchange_name}")
        print(f"Exchange config: {exchange_config}")
        print(f"Exchange attributes:")
        for key, value in exchange.__dict__.items():
            print(f"{key}: {value}")
        print()

    if not exchange.check_required_credentials():
        print(f"Error: Unable to authenticate with {exchange_name}.")
        exit()

    return exchange


def fetch_latest_candles(exchange, symbol, timeframe, limit):
    """
    Fetch the latest candles from the exchange.
    """
    candles = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    return np.array(candles)


def fetch_candles(exchange, symbol, timeframe, since, limit=1000):
    """
    Fetch a batch of candles from the exchange.
    """
    return exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)


def fetch_all_candles(exchange, symbol, timeframe, target_count=10000):
    """
    Fetch multiple batches of candles until reaching the target count.
    """
    all_candles = []
    timeframe_in_ms = timeframe_to_milliseconds(timeframe)
    current_time = exchange.milliseconds()
    since = current_time  # Ensure 'since' is an integer
    fetched_count = 0

    prev_candles_batch_0 = 0
    prev_candles_batch_min1 = 0

    print(f"\nFetching all candles for {symbol} on {exchange.id}:")
    while fetched_count < target_count:
        batch_size = min(target_count - fetched_count, 1000)
        since_param = (
            int(since - timeframe_in_ms * batch_size) if since else None
        )  # Adjust 'since' for each call
        candles_batch = fetch_candles(
            exchange, symbol, timeframe, since_param, batch_size
        )

        if not candles_batch:
            break

        all_candles.extend(candles_batch)
        fetched_count += len(candles_batch)
        since = candles_batch[0][
            0
        ]  # Update 'since' to the oldest timestamp from the batch

        # Print the timestamp of the first and last candle in the batch in human-readable form
        print(
            f"* {len(candles_batch)} candles, {target_count - fetched_count} remaining."
        )
        print(
            f"  Oldest: {timestamp_to_human_readable(candles_batch[0][0])} - Newest: {timestamp_to_human_readable(candles_batch[-1][0])}"
        )

        if (
            candles_batch[0][0] == prev_candles_batch_0
            or candles_batch[-1][0] == prev_candles_batch_min1
        ):
            if DEBUG_MODE:
                print(
                    f"Candles batch is identical to the previous batch. Exiting loop."
                )
            break

        prev_candles_batch_0 = candles_batch[0][0]
        prev_candles_batch_min1 = candles_batch[-1][0]

        # Respect the exchange rate limit
        time.sleep(exchange.rateLimit / 1000)

    # Fetch the latest few candles separately
    latest_candles = fetch_latest_candles(exchange, symbol, timeframe, 1000)
    # Assuming all_candles and latest_candles are NumPy arrays
    all_candles = np.concatenate((all_candles, latest_candles), axis=0)

    # Deduplicate and sort all candles by timestamp in ascending order
    unique_candles = list({c[0]: c for c in all_candles}.values())
    unique_candles.sort(key=lambda x: x[0])

    if len(unique_candles) > target_count:
        unique_candles = unique_candles[-target_count:]

    print(f"\nTotal unique candles fetched: {len(unique_candles)}")

    # Remove newest candle (if applied, script wont work)
    # if len(unique_candles) > 1:
    #     unique_candles = unique_candles[:-1]

    # Reverse order
    unique_candles = unique_candles[::-1]

    candles_to_return = np.array(unique_candles)

    print_first_last_candle_timestamps(candles_to_return)

    return candles_to_return


def timeframe_to_milliseconds(timeframe):
    """
    Convert a timeframe string to milliseconds.
    """
    milliseconds = {
        "1m": 60000,
        "5m": 300000,
        "15m": 900000,
        "30m": 1800000,
        "1h": 3600000,
        "4h": 14400000,
        "1d": 86400000,
    }
    return milliseconds[timeframe]


def timeframe_to_seconds(timeframe):
    """
    Convert a timeframe string to seconds.
    """
    seconds = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "4h": 14400,
        "1d": 86400,
    }
    return seconds[timeframe]


def validate_candles_data(candles, timeframe):
    """
    Validate the integrity of the candles data.

    :param candles: numpy array of candles data
    :param timeframe: string timeframe used for fetching candles (e.g., '1h', '5m', '1d')
    :return: bool indicating if the candles data is valid
    """
    # Convert timeframe to seconds to compute gaps
    seconds_per_candle = timeframe_to_seconds(timeframe)

    # Check if candles are sorted in reverse chronological order
    if not all(candles[i][0] >= candles[i + 1][0] for i in range(len(candles) - 1)):
        print("Candles are not sorted in reverse chronological order.")
        return False

    # Check for duplicate candles
    timestamps = candles[:, 0]
    if len(timestamps) != len(set(timestamps)):
        print("Duplicate candles detected.")
        return False

    # Check for gaps in the candles
    # if any(candles[i][0] - candles[i+1][0] != seconds_per_candle * 1000 for i in range(len(candles) - 1)):
    #     print("Gap found between candles.")
    #     return False

    # Check the timestamp of the latest candle (should be within one timeframe interval of the current time)
    current_time = int(time.time() * 1000)  # Current time in milliseconds
    if current_time - candles[0][0] > seconds_per_candle * 1000:
        print("The latest candle is too old.")
        return False

    # All checks passed
    print("Candle data integrity validated successfully.")
    return True

def compute_scores(candles, perc, nbr):
    """
    Compute the scores for the given candles which are sorted from oldest to newest (reverse chronological order).
    """
    opens = candles[:, 1]
    highs = candles[:, 2]
    lows = candles[:, 3]
    closes = candles[:, 4]

    num_candles = len(candles)
    step_sizes = [closes[-1] * (perc / 100) * (i + 1) for i in range(nbr)]

    # Exponential decay factor
    decay_factor = 0.95
    weights = np.array([decay_factor ** (num_candles - i) for i in range(1, num_candles + 1)])

    green_high_hits = [0] * nbr
    green_low_hits = [0] * nbr
    red_high_hits = [0] * nbr
    red_low_hits = [0] * nbr

    green_high_probs = [0] * nbr
    green_low_probs = [0] * nbr
    red_high_probs = [0] * nbr
    red_low_probs = [0] * nbr

    total_weighted_green = total_weighted_red = 0

    for i in range(num_candles):
        green = closes[i] > opens[i]
        red = not green
        weight = weights[i]

        # Apply weight to the count
        total_weighted_green += green * weight
        total_weighted_red += red * weight

        for level, step in enumerate(step_sizes):
            if green and highs[i] >= highs[i - 1] + step:
                green_high_hits[level] += weight
            if green and lows[i] <= lows[i - 1] - step:
                green_low_hits[level] += weight
            if red and highs[i] >= highs[i - 1] + step:
                red_high_hits[level] += weight
            if red and lows[i] <= lows[i - 1] - step:
                red_low_hits[level] += weight

    # Convert hit counts into probabilities, accounting for the total weighted counts
    for i in range(nbr):
        if total_weighted_green > 0:
            green_high_probs[i] = (green_high_hits[i] / total_weighted_green) * 100
            green_low_probs[i] = (green_low_hits[i] / total_weighted_green) * 100
        if total_weighted_red > 0:
            red_high_probs[i] = (red_high_hits[i] / total_weighted_red) * 100
            red_low_probs[i] = (red_low_hits[i] / total_weighted_red) * 100

    return (
        green_high_hits,
        green_low_hits,
        red_high_hits,
        red_low_hits,
        green_high_probs,
        green_low_probs,
        red_high_probs,
        red_low_probs,
        total_weighted_green,
        total_weighted_red,
    )

def calculate_probabilities(
    current_price,
    green_high_probs,
    green_low_probs,
    red_high_probs,
    red_low_probs,
    closes,
    perc,
    nbr,
):
    """
    Calculate the probabilities for the given candles which are sorted from oldest to newest (reverse chronological order).
    """
    probabilities = []
    step_sizes = [closes[-1] * (perc / 100) * (i + 1) for i in range(nbr)]

    for i in range(nbr):
        step = step_sizes[i]
        high_price_level = closes[-1] + step
        low_price_level = closes[-1] - step

        if green_high_probs[i] > 0:
            long_or_short = "LONG" if high_price_level > current_price else "SHORT"
            distance_in_pct = round(
                (abs(high_price_level - current_price) / current_price) * 100, 2
            )
            probabilities.append(
                (green_high_probs[i], high_price_level, long_or_short, distance_in_pct)
            )
        if green_low_probs[i] > 0:
            long_or_short = "LONG" if low_price_level > current_price else "SHORT"
            distance_in_pct = round(
                (abs(low_price_level - current_price) / current_price) * 100, 2
            )
            probabilities.append(
                (green_low_probs[i], low_price_level, long_or_short, distance_in_pct)
            )
        if red_high_probs[i] > 0:
            long_or_short = "LONG" if high_price_level > current_price else "SHORT"
            distance_in_pct = round(
                (abs(high_price_level - current_price) / current_price) * 100, 2
            )
            probabilities.append(
                (red_high_probs[i], high_price_level, long_or_short, distance_in_pct)
            )
        if red_low_probs[i] > 0:
            long_or_short = "LONG" if low_price_level > current_price else "SHORT"
            distance_in_pct = round(
                (abs(low_price_level - current_price) / current_price * 100), 2
            )
            probabilities.append(
                (red_low_probs[i], low_price_level, long_or_short, distance_in_pct)
            )

    # Aggregating probabilities for the same price levels
    aggregated_probabilities = {}
    for prob, price, long_or_short, distance_in_pct in probabilities:
        key = (price, long_or_short, distance_in_pct)  # Unique key for each combination
        if key not in aggregated_probabilities:
            # print(f"Skip sum - {prob}% - {price}")
            aggregated_probabilities[key] = prob
        else:
            # print(f"Sum - {prob}% - {price}")
            aggregated_probabilities[key] += prob  # Summing probabilities for the same price level

    # Convert the aggregated results back to a list, maintaining the original structure
    final_probabilities = [(prob, price, long_or_short, distance_in_pct) for (price, long_or_short, distance_in_pct), prob in aggregated_probabilities.items()]

    # Remove 0% chance probabilities and sort
    final_probabilities = [p for p in final_probabilities if p[0] > 0]

    # Sort descending by probability
    final_probabilities.sort(reverse=True, key=lambda x: x[0])

    return final_probabilities


def timestamp_to_human_readable(timestamp_ms):
    """
    Converts Unix timestamp in milliseconds to a human-readable date-time string.
    """
    return datetime.utcfromtimestamp(timestamp_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")


def print_first_last_candle_timestamps(candles):
    """
    Print the first and last candle timestamps.
    """
    # Convert the timestamp from milliseconds to seconds for datetime
    first_candle_timestamp = candles[0, 0] / 1000
    last_candle_timestamp = candles[-1, 0] / 1000

    # Convert to human-readable format
    first_candle_datetime = datetime.utcfromtimestamp(first_candle_timestamp).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    last_candle_datetime = datetime.utcfromtimestamp(last_candle_timestamp).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    first_candle_open_price = candles[0, 1]
    last_candle_open_price = candles[-1, 1]
    first_candle_close_price = candles[0, 4]
    last_candle_close_price = candles[-1, 4]

    print()
    print(f"Oldest candle timestamp: {last_candle_datetime}")
    print(f"Oldest candle open price: {last_candle_open_price}")
    print(f"Oldest candle close price: {last_candle_close_price}")
    print()
    print(f"Newest candle timestamp: {first_candle_datetime}")
    print(f"Newest candle open price: {first_candle_open_price}")
    print(f"Newest candle close price: {first_candle_close_price}\n")


def fetch_current_price(exchange, symbol):
    """
    Fetch the current price of the symbol.
    """
    ticker = exchange.fetch_ticker(symbol)
    print(f"\nCurrent price of {symbol}: {ticker['last']}")
    log_message(exchange.name, symbol, f"Current price of {symbol}: {ticker['last']}")
    return ticker["last"]  # Using the last price as the current price


def place_trade_with_dynamic_stop_loss(
    exchange, symbol, probabilities, current_price, long_confidence, short_confidence
):
    MIN_CONFIDENCE = 7
    MIN_PROBABILITY = 30
    MIN_DISTANCE = 0.005

    highest_prob_trade = None
    for prob, price, long_or_short, distance_in_pct in probabilities:
        if (
            price > current_price
            and long_confidence > MIN_CONFIDENCE
            and prob > MIN_PROBABILITY
            and distance_in_pct > MIN_DISTANCE
        ):
            highest_prob_trade = ("LONG", price, prob)
            break
        elif (
            price < current_price
            and short_confidence > MIN_CONFIDENCE
            and prob > MIN_PROBABILITY
            and distance_in_pct > MIN_DISTANCE
        ):
            highest_prob_trade = ("SHORT", price, prob)
            break

    if highest_prob_trade:
        direction, price_level, probability = highest_prob_trade
        print(
            f"Initiating {direction} trade with {probability:.2f}% probability at price level {price_level}"
        )

        # Calculate stop-loss based on the lowest probabilistic opposite side price level
        opposite_direction_prices = [
            price
            for prob, price, long_or_short, distance_in_pct in probabilities
            if (direction == "LONG" and price < current_price)
            or (direction == "SHORT" and price > current_price)
        ]
        if opposite_direction_prices:
            if direction == "LONG":
                stop_loss_price = min(
                    opposite_direction_prices
                )  # For LONG, use the highest of the lower prices
            else:
                stop_loss_price = max(
                    opposite_direction_prices
                )  # For SHORT, use the lowest of the higher prices

            # Enforce maximum stop-loss distance rule (double the distance to target)
            distance_to_target = abs(price_level - current_price)
            max_stop_loss_distance = distance_to_target * 5  # 5x distance to target
            if abs(stop_loss_price - current_price) > max_stop_loss_distance:
                if direction == "LONG":
                    stop_loss_price = round(
                        current_price - max_stop_loss_distance,
                        len(str(current_price).split(".")[1]),
                    )
                else:
                    stop_loss_price = round(
                        current_price + max_stop_loss_distance,
                        len(str(current_price).split(".")[1]),
                    )
        else:
            # Fallback in case no opposite direction prices found
            stop_loss_price = (
                current_price - distance_to_target
                if direction == "LONG"
                else current_price + distance_to_target
            )

        take_profit_price = price_level

        # Convert this to actual order logic using Binance API
        # Placeholder for demonstration
        print(
            f"Place limit order for {direction} at {current_price}, stop-loss at {stop_loss_price} and take-profit at {take_profit_price}"
        )
        # Actual Binance API calls would go here
    else:
        print("No trade meets the criteria")


def calculate_atr(candles, period=4):
    high = candles[:, 2]
    low = candles[:, 3]
    close = candles[:, 4]

    # Prepend the first close value to close[:-1] to match the shape with high and low
    # This assumes the first transaction is a valid representation of the previous close
    previous_close = np.roll(close, shift=1)
    previous_close[0] = close[0]  # Adjust this line as needed for your logic

    tr = np.maximum(
        high - low, np.maximum(abs(high - previous_close), abs(low - previous_close))
    )
    atr = np.mean(
        tr[:period]
    )  # Calculate the average of the True Range over the specified period

    print(f"\nAverage True Range (ATR, {period}): {atr}.\n")

    return atr


def main(config):
    """
    Main function to run the script.
    """
    # Load reset variables
    lookback_candles = config["defaults"]["breakoutLookbackCdls"]

    # User input for symbol
    exchange_input = input(f"Enter the exchange name [{DEFAULT_EXCHANGE}]: ").lower()
    exchange_name = exchange_input if exchange_input else DEFAULT_EXCHANGE
    exchange = initialize_exchange(exchange_name, config)
    symbol_input = input(f"Enter the symbol [{DEFAULT_SYMBOL}]: ").upper()
    symbol = symbol_input if symbol_input else DEFAULT_SYMBOL
    timeframe_input = input(f"Enter the timeframe [{DEFAULT_TIMEFRAME}]: ").lower()
    timeframe = timeframe_input if timeframe_input else DEFAULT_TIMEFRAME
    nbr_input = input(
        f"Enter the number of breakout levels [{DEFAULT_NBR_LEVELS}]: "
    ).lower()
    nbr = int(nbr_input) if nbr_input else DEFAULT_NBR_LEVELS

    # Fetch candles (0 is newest)
    candles = fetch_all_candles(exchange, symbol, timeframe, lookback_candles)
    if (
        candles is None
        or len(candles) == 0
        or not validate_candles_data(candles, timeframe)
    ):
        print(
            f"Error: Unable to fetch or validate candles for {symbol} on {exchange.name}."
        )
        log_message(
            exchange.name, symbol, "Error: Unable to fetch or validate candles."
        )
        exit()

    # Step 0: Reverse Candles (0 is oldest)
    reverse_candles = candles[::-1]

    # Fetch current price
    current_price = fetch_current_price(exchange, symbol)

    # Fetch Current ATR
    atr_value = calculate_atr(candles, 4)

    # Calculate minimum percentage between breakout levels
    DEFAULT_MIN_PCT = (
        float(round((atr_value / current_price) * 100 / (nbr * 2), 2))
        if atr_value > 0
        else DEFAULT_MIN_PCT
    )

    # User input for minimum percentage between breakout levels
    perc_input = input(
        f"Enter the minimum percentage between breakout levels [{DEFAULT_MIN_PCT}]: "
    ).lower()
    perc = float(perc_input) if perc_input else DEFAULT_MIN_PCT
    perc = perc if perc > 0 else DEFAULT_MIN_PCT

    # Step 1: Compute Scores
    (
        green_high_hits,
        green_low_hits,
        red_high_hits,
        red_low_hits,
        green_high_probs,
        green_low_probs,
        red_high_probs,
        red_low_probs,
        total_green,
        total_red,
    ) = compute_scores(reverse_candles, perc, nbr)

    # Step 2: Calculate Probabilities
    probabilities = calculate_probabilities(
        current_price,
        green_high_probs,
        green_low_probs,
        red_high_probs,
        red_low_probs,
        reverse_candles[:, 4],
        perc,
        nbr,
    )

    # Step 3: Print Results
    log_message(
        exchange.name,
        symbol,
        f"Breakout Levels - {exchange.name} - {symbol} Probabilities Plan:",
    )
    print(f"\nBreakout Levels - {exchange.name} - {symbol} Probabilities Plan:")
    print("-----------------------------------")
    print(f"Exchange: {exchange.name}")
    print(f"Symbol: {symbol}")
    print(f"Current Price: {current_price}")
    print(f"Open Price: {candles[0][1]}")
    print(f"Close Price: {candles[0][4]}")
    print(f"Timeframe: {timeframe}")
    print(f"ATR: {atr_value}")
    print(f"# of Candles: {len(candles)}")
    print(f"# of Breakout Levels: {nbr}")
    print(f"Min. % for Breakout Levels: {perc}%")
    if DEBUG_MODE:
        print(f"Total Green: {total_green}")
        print(f"Total Red: {total_red}")
        print(f"Green High Probabilities: {green_high_probs}")
        print(f"Green Low Probabilities: {green_low_probs}")
        print(f"Red High Probabilities: {red_high_probs}")
        print(f"Red Low Probabilities: {red_low_probs}")
        print(f"Green High Hits: {green_high_hits}")
        print(f"Green Low Hits: {green_low_hits}")
        print(f"Red High Hits: {red_high_hits}")
        print(f"Red Low Hits: {red_low_hits}")
    print(f"-----------------------------------")
    for i, (prob, price, long_or_short, distance_in_pct) in enumerate(
        probabilities, start=1
    ):
        print(
            f"{i}. Probability: {prob:.2f}% - Price Level: {price} - Direction: {long_or_short} - Distance: {distance_in_pct}%"
        )
        log_message(
            exchange.name,
            symbol,
            f"{i}. Probability: {prob:.2f}% - Price Level: {price} - Direction: {long_or_short} - Distance: {distance_in_pct}%",
        )

    # Step 4: Calculate all long and short probabilities:
    # Calculate all long probabilities
    all_long_probabilities = 0
    for prob, price, long_or_short, distance_in_pct in probabilities:
        if long_or_short == "LONG":
            all_long_probabilities += prob

    # Calculate all short probabilities
    all_short_probabilities = 0
    for prob, price, long_or_short, distance_in_pct in probabilities:
        if long_or_short == "SHORT":
            all_short_probabilities += prob

    # Calculate long confidence by dividing all_long_probabilities by all_short_probabilities and vice versa
    long_confidence = ((all_long_probabilities / all_short_probabilities) - 1) * 100
    short_confidence = ((all_short_probabilities / all_long_probabilities) - 1) * 100

    # Print results
    print(f"\nSum of LONG probabilities: {all_long_probabilities:.2f}%")
    log_message(
        exchange.name, symbol, f"Sum of LONG probabilities: {all_long_probabilities:.2f}%"
    )
    print(f"LONG confidence: {long_confidence:.2f}")
    log_message(exchange.name, symbol, f"LONG confidence: {long_confidence:.2f}")

    print(f"Sum of SHORT probabilities: {all_short_probabilities:.2f}%")
    log_message(
        exchange.name,
        symbol,
        f"Sum of SHORT probabilities: {all_short_probabilities:.2f}%",
    )
    print(f"SHORT confidence: {short_confidence:.2f}\n")
    log_message(exchange.name, symbol, f"SHORT confidence: {short_confidence:.2f}")

    # Step 5: Execute Trades
    place_trade_with_dynamic_stop_loss(
        exchange,
        symbol,
        probabilities,
        current_price,
        long_confidence,
        short_confidence,
    )

    # Step 6: Exit
    print("\nDone.")
    log_message(exchange.name, symbol, "--- THE END ---")


if __name__ == "__main__":
    # Load API configuration
    config = load_config()

    # Load default variables
    DEBUG_MODE = config["debug"]["enabled"]
    if DEBUG_MODE:
        print("Debug mode is enabled.\n")
    else:
        DEBUG_MODE = False
        print("Debug mode is disabled.\n")

    DEFAULT_EXCHANGE = config["defaults"]["exchange"]
    if DEFAULT_EXCHANGE:
        print(f"Default exchange is {DEFAULT_EXCHANGE}.\n")
    else:
        DEFAULT_EXCHANGE = "binance"

    DEFAULT_SYMBOL = config["defaults"]["symbol"]
    if DEFAULT_SYMBOL:
        print(f"Default symbol is {DEFAULT_SYMBOL}.\n")
    else:
        DEFAULT_SYMBOL = "BTC/USDT"

    DEFAULT_TIMEFRAME = config["defaults"]["timeframe"]
    if DEFAULT_TIMEFRAME:
        print(f"Default timeframe is {DEFAULT_TIMEFRAME}.\n")
    else:
        DEFAULT_TIMEFRAME = "1d"

    DEFAULT_MIN_PCT = float(config["defaults"]["breakoutMinPct"])
    if DEFAULT_MIN_PCT:
        print(
            f"Default minimum percentage between breakout levels is {DEFAULT_MIN_PCT}.\n"
        )
    else:
        DEFAULT_MIN_PCT = 0.25

    DEFAULT_NBR_LEVELS = int(config["defaults"]["breakoutNumLvls"])
    if DEFAULT_NBR_LEVELS:
        print(f"Default number of breakout levels is {DEFAULT_NBR_LEVELS}.\n")
    else:
        DEFAULT_NBR_LEVELS = 5

    main(config)
