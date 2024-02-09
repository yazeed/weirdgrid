import matplotlib
matplotlib.use('Agg')  # Use the Agg backend which does not require Tk
import matplotlib.pyplot as plt

def visualize_orders(orders, current_price):
    """
    Visualize the distribution of orders with a plot.
    """
    buy_orders = [(price, qty) for price, qty in orders if price < current_price]
    sell_orders = [(price, qty) for price, qty in orders if price > current_price]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    if buy_orders:
        buy_prices, buy_quantities = zip(*buy_orders)
        plt.scatter(buy_prices, buy_quantities, color='green', label='Buy Orders')
    if sell_orders:
        sell_prices, sell_quantities = zip(*sell_orders)
        plt.scatter(sell_prices, sell_quantities, color='red', label='Sell Orders')
    plt.axvline(x=current_price, color='blue', label='Current Price')
    plt.title('Distribution of Orders')
    plt.xlabel('Price')
    plt.ylabel('Quantity')
    plt.legend()
    plt.show()
    plt.savefig('orders.png')

Visualize orders
visualize_orders(orders, current_price)
