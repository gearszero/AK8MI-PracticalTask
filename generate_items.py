import numpy as np


# Generování dat pro KP a MCKP
def generate_knapsack_items(num_items, price_range=(1, 50), weight_range=(1, 50)):
    prices = np.random.randint(price_range[0], price_range[1] + 1, size=num_items)
    weights = np.random.randint(weight_range[0], weight_range[1] + 1, size=num_items)
    return prices, weights


def generate_mckp_items(num_classes, items_per_class=3, price_range=(1, 50), weight_range=(1, 50)):
    prices = np.random.randint(price_range[0], price_range[1] + 1, size=(num_classes, items_per_class))
    weights = np.random.randint(weight_range[0], weight_range[1] + 1, size=(num_classes, items_per_class))
    return prices, weights
