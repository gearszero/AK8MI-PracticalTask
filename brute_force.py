from itertools import combinations, product


# Brute force pro KP
def brute_force_knapsack(prices, weights, capacity):
    num_items = len(prices)
    best_value = 0
    best_combination = []
    max_fes = 0
    values = []

    # Projít všechny možné kombinace předmětů
    for r in range(1, num_items + 1):
        for combination in combinations(range(num_items), r):
            max_fes += 1
            total_weight = sum(weights[i] for i in combination)
            total_price = sum(prices[i] for i in combination)
            if total_weight <= capacity and total_price > best_value:
                best_value = total_price
                best_combination = combination
            values.append(best_value)

    return best_combination, best_value, max_fes, values


# Brute force pro MCKP
def brute_force_mckp(prices, weights, capacity):
    num_classes, items_per_class = prices.shape
    best_value = 0
    best_combination = None
    maxFES = 0
    values = []

    for combination in product(*[range(items_per_class) for _ in range(num_classes)]):
        maxFES += 1
        total_weight = sum(weights[i][combination[i]] for i in range(num_classes))
        total_price = sum(prices[i][combination[i]] for i in range(num_classes))
        if total_weight <= capacity and total_price > best_value:
            best_value = total_price
            best_combination = combination
        values.append(best_value)

    # Ošetření případu, kdy není nalezeno žádné platné řešení
    if best_combination is None:
        best_combination = []

    return best_combination, best_value, maxFES, values


