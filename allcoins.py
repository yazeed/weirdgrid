import requests
import time
import sqlite3

# market_cap_to_supply_price_ratio (simple market cap elasticity) = market cap / (circulating supply * price)
## higher is overpriced (high sensitivity to future inflation), lower is underpriced (less sensitivity to future inflation)
### supply focused

# supply_adjustment_factor (advanced market cap elasticity) = market_cap_to_supply_price_ratio * (circulating supply / total supply) * (circulating supply / max supply)
## higher is indicative of entering price discovery due to scarcity of supply, lower is indicative of more price stability due to high supply availability
### demand focused

# crypto_multiplier = (circulating_supply / (circulating_supply - (total_supply - circulating_supply))) * (circulating_supply / market_cap)
## higher is indicative of entering price discovery due to scarcity of supply, lower is indicative of more price stability due to high supply availability
### demand focused

# dilution_growth_potential = fully diluted market cap / market cap
## higher is risky due to high dilution potential, lower is more stable and less risky
### long term supply focused

def create_database():
    conn = sqlite3.connect('cryptocurrencies.db')
    c = conn.cursor()
    # Create the cryptocurrencies table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS cryptocurrencies
                 (id TEXT PRIMARY KEY, symbol TEXT, name TEXT, market_cap_rank REAL, market_cap_category TEXT, current_price REAL DEFAULT 0, market_cap REAL DEFAULT 0, 
                  fully_diluted_valuation REAL DEFAULT 0, total_volume REAL DEFAULT 0, circulating_supply REAL DEFAULT 0, 
                  total_supply REAL DEFAULT 0, max_supply REAL DEFAULT 0, ath REAL DEFAULT 0, atl REAL DEFAULT 0, market_cap_to_supply_price_ratio REAL DEFAULT 0,
                  supply_adjustment_factor REAL DEFAULT 0, crypto_multiplier REAL DEFAULT 0, price_sensitivity REAL DEFAULT 0, dilution_growth_potential REAL DEFAULT 0,
                  binance BOOLEAN DEFAULT 0, kucoin BOOLEAN DEFAULT 0, okx BOOLEAN DEFAULT 0, mexc BOOLEAN DEFAULT 0);''')
    # Create the markets table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS markets
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, coin_id TEXT, exchange TEXT, trust_score TEXT,
                  pair TEXT, price REAL DEFAULT 0, volume REAL DEFAULT 0, bid_ask_spread_percentage REAL DEFAULT 0, UNIQUE(coin_id, exchange, pair));''')
    conn.commit()
    conn.close()

# ?x_cg_demo_api_key=CG-QGvDpY7Cza5VPjJnPDTvHdH5
def fetch_cryptocurrencies(page=1, retries=5, retry_delay=60):
    url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=250&page={page}&sparkline=false&x_cg_demo_api_key=CG-QGvDpY7Cza5VPjJnPDTvHdH5"
    
    for attempt in range(retries):
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            print("Rate limit exceeded. Waiting to retry...")
            time.sleep(retry_delay)  # Wait for retry_delay seconds before retrying
        else:
            print(f"Failed to fetch data: {response.status_code}")
            return []
    
    print("Max retries exceeded. Please try again later.")
    return []

# ?x_cg_demo_api_key=CG-QGvDpY7Cza5VPjJnPDTvHdH5
def get_market_data(coin_id, retries=5, retry_delay=60):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/tickers?x_cg_demo_api_key=CG-QGvDpY7Cza5VPjJnPDTvHdH5"
    market_data = []

    for attempt in range(retries):
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            tickers = data.get('tickers', [])
            for ticker in tickers:
                # Filter for specific exchanges if necessary, e.g., Binance, KuCoin
                # Example: if ticker['market']['name'] in ['Binance', 'KuCoin']:
                market_info = {
                    'coin_id': coin_id,
                    'exchange': ticker['market']['name'],
                    'pair': ticker['base'] + '/' + ticker['target'],
                    'price': ticker['last'],
                    'trust_score': ticker['trust_score'],
                    'volume': ticker['volume'],
                    'bid_ask_spread_percentage': ticker['bid_ask_spread_percentage'],
                }
                market_data.append(market_info)
            return market_data
        elif response.status_code == 429:
            print(f"Rate limit exceeded while fetching market data for {coin_id}. Waiting to retry...")
            time.sleep(retry_delay)  # Wait for retry_delay seconds before retrying
        else:
            print(f"Failed to fetch market data for {coin_id}: {response.status_code}")
            return market_data

    print(f"Max retries exceeded for {coin_id}. Please try again later.")
    return market_data

def update_database(cryptocurrencies):
    conn = sqlite3.connect('cryptocurrencies.db')
    c = conn.cursor()
    for crypto in cryptocurrencies:
        # Ensure numeric fields are set to 0 if they are missing
        id = crypto['id']
        symbol = crypto['symbol']
        name = crypto['name']
        market_cap_rank = crypto.get('market_cap_rank', 0) or 0
        current_price = crypto.get('current_price', 0) or 0
        market_cap = crypto.get('market_cap', 0) or 0
        if market_cap > 10000000000:
            market_cap_category = '(A) above 10B'
        elif market_cap > 2000000000:
            market_cap_category = '(B) above 2B'
        elif market_cap > 400000000:
            market_cap_category = '(C) above 400M'
        elif market_cap > 80000000:
            market_cap_category = '(D) above 80M'
        elif market_cap > 20000000:
            market_cap_category = '(E) above 20M'
        elif market_cap > 4000000:
            market_cap_category = '(F) above 4M'
        elif market_cap > 800000:
            market_cap_category = '(G) above 800K'
        elif market_cap > 200000:
            market_cap_category = '(H) above 200K'
        elif market_cap > 40000:
            market_cap_category = '(I) above 40K'
        else:
            market_cap_category = '(J) below 40K'
        fully_diluted_valuation = crypto.get('fully_diluted_valuation', 0) or 0
        total_volume = crypto.get('total_volume', 0) or 0
        circulating_supply = crypto.get('circulating_supply', 0) or 0
        total_supply = crypto.get('total_supply', 0) or 0
        max_supply = crypto.get('max_supply', 0) or 0
        ath = crypto.get('ath', 0) or 0
        atl = crypto.get('atl', 0) or 0

        # Calculate additional metrics
        market_cap_to_supply_price_ratio = market_cap / (circulating_supply * current_price) if circulating_supply and current_price else 0
        supply_adjustment_factor = market_cap_to_supply_price_ratio * (circulating_supply / total_supply) * (circulating_supply / max_supply) if total_supply and max_supply and market_cap_to_supply_price_ratio else 0
        crypto_multiplier = 1 + (circulating_supply / (circulating_supply - (total_supply - circulating_supply))) * (circulating_supply / market_cap) if total_supply and market_cap and circulating_supply - (total_supply - circulating_supply) != 0 else 0
        price_sensitivity = (market_cap / circulating_supply) / (fully_diluted_valuation / max_supply) if circulating_supply and max_supply else 0
        dilution_growth_potential = 1 - (market_cap / fully_diluted_valuation) if fully_diluted_valuation else 0

        # Update market data and determine which exchanges the cryptocurrency is traded on
        markets = get_market_data(crypto['id'])
        binance, kucoin, okx, mexc = False, False, False, False  # Initialize exchange presence flags
        for market in markets:
            exchange = market['exchange'].lower()
            if 'binance' in exchange:
                binance = True
            if 'kucoin' in exchange:
                kucoin = True
            if 'okx' in exchange:
                okx = True
            if 'mexc' in exchange or 'mxc' in exchange:  # Assuming MXC refers to MEXC as well
                mexc = True

        # Prepare the data tuple
        data_tuple = (id, symbol, name, market_cap_rank, market_cap_category, current_price, market_cap, fully_diluted_valuation, 
                      total_volume, circulating_supply, total_supply, max_supply, ath, atl,
                      market_cap_to_supply_price_ratio, supply_adjustment_factor, crypto_multiplier, price_sensitivity, dilution_growth_potential,
                      binance, kucoin, okx, mexc)

        print(f"Updating {symbol} ({id})...")
        # Insert or update the cryptocurrency information
        c.execute('''INSERT INTO cryptocurrencies
                     (id, symbol, name, market_cap_rank, market_cap_category, current_price, market_cap, 
                      fully_diluted_valuation, total_volume, circulating_supply, total_supply, max_supply,
                      ath, atl, market_cap_to_supply_price_ratio, supply_adjustment_factor, crypto_multiplier, price_sensitivity,
                      dilution_growth_potential, binance, kucoin, okx, mexc)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                     ON CONFLICT(id) DO UPDATE SET
                     symbol=excluded.symbol, name=excluded.name, market_cap_rank=excluded.market_cap_rank, 
                     market_cap_category=excluded.market_cap_category, current_price=excluded.current_price, market_cap=excluded.market_cap, 
                     fully_diluted_valuation=excluded.fully_diluted_valuation, total_volume=excluded.total_volume, 
                     circulating_supply=excluded.circulating_supply, total_supply=excluded.total_supply, 
                     max_supply=excluded.max_supply, ath=excluded.ath, atl=excluded.atl, market_cap_to_supply_price_ratio=excluded.market_cap_to_supply_price_ratio,
                     supply_adjustment_factor=excluded.supply_adjustment_factor, crypto_multiplier=excluded.crypto_multiplier, price_sensitivity=excluded.price_sensitivity,
                     dilution_growth_potential=excluded.dilution_growth_potential,
                     binance=excluded.binance, kucoin=excluded.kucoin, okx=excluded.okx, mexc=excluded.mexc''', 
                     data_tuple)

        # Update market data
        for market in markets:
            print(f"Updating {symbol} ({id}) market data for {market['exchange']} {market['pair']}...")
            # Insert or update the market information
            c.execute('''INSERT INTO markets (coin_id, exchange, trust_score, pair, price, volume, bid_ask_spread_percentage)
                         VALUES (?, ?, ?, ?, ?, ?, ?)
                         ON CONFLICT(coin_id, exchange, pair) DO UPDATE SET
                         trust_score=excluded.trust_score, price=excluded.price, volume=excluded.volume, bid_ask_spread_percentage=excluded.bid_ask_spread_percentage''',
                      (market['coin_id'], market['exchange'], market['trust_score'], market['pair'], market['price'], market['volume'], market['bid_ask_spread_percentage']))
            
        conn.commit()
    
    conn.close()

def main():
    create_database()

    page = 1
    while True:
        cryptocurrencies = fetch_cryptocurrencies(page=page)
        if not cryptocurrencies:
            break  # Stop if no more cryptocurrencies are fetched
        # try:
        update_database(cryptocurrencies)
        print(f"Page {page} processed successfully.")
        page += 1
        time.sleep(1)  # Throttle requests to avoid hitting rate limits
        # except Exception as e:
        #     print(f"Page {page} failed to process.", e)
        #     next

if __name__ == "__main__":
    main()