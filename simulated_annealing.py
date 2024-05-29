import random
import numpy as np


def knapsack_value(weights, prices, solution):
    total_weight = np.sum(weights * solution)
    total_price = np.sum(prices * solution)
    return total_price, total_weight


def generate_neighbor_solution(solution, num_items):
    neighbor = solution.copy()
    index = random.randint(0, num_items - 1)
    neighbor[index] = 1 - neighbor[index]  # Flip the bit
    return neighbor


def simulated_annealing_knapsack(weights, prices, capacity, initial_temp, final_temp, cooling_rate, max_iter, maxFES):
    num_items = len(weights)

    # Inicializace počátečního řešení, které splňuje kapacitní omezení
    while True:
        current_solution = np.random.randint(0, 2, num_items)
        current_value, current_weight = knapsack_value(weights, prices, current_solution)
        if current_weight <= capacity:
            break

    best_solution = current_solution.copy()
    best_value = current_value
    best_weight = current_weight

    temp = initial_temp
    iteration = 0
    values = []
    fes = 0

    while temp > final_temp and iteration < max_iter and fes < maxFES:
        neighbor_solution = generate_neighbor_solution(current_solution, num_items)
        neighbor_value, neighbor_weight = knapsack_value(weights, prices, neighbor_solution)
        fes += 1

        if neighbor_weight <= capacity and (neighbor_value > current_value or random.uniform(0, 1) < np.exp((neighbor_value - current_value) / temp)):
            current_solution = neighbor_solution
            current_value = neighbor_value
            current_weight = neighbor_weight

            if current_value > best_value:
                best_solution = current_solution.copy()
                best_value = current_value
                best_weight = current_weight

        values.append(best_value)
        temp *= cooling_rate
        iteration += 1

    return best_solution, best_value, values, best_weight, fes


def mckp_value(weights, prices, solution):
    total_weight = sum(weights[i][solution[i]] for i in range(len(solution)))
    total_price = sum(prices[i][solution[i]] for i in range(len(solution)))
    return total_price, total_weight


def generate_neighbor_solution_mckp(solution, num_classes, items_per_class):
    neighbor = solution.copy()
    while True:
        index = random.randint(0, num_classes - 1)
        new_value = random.randint(0, items_per_class - 1)
        if new_value != neighbor[index]:
            neighbor[index] = new_value
            break
    return neighbor


def simulated_annealing_mckp(weights, prices, capacity, initial_temp, final_temp, cooling_rate, max_iter, maxFES):
    num_classes, items_per_class = weights.shape

    while True:
        current_solution = [random.randint(0, items_per_class - 1) for _ in range(num_classes)]
        current_value, current_weight = mckp_value(weights, prices, current_solution)
        if current_weight <= capacity:
            break

    best_solution = current_solution.copy()
    best_value = current_value
    best_weight = current_weight

    temp = initial_temp
    iteration = 0
    values = []
    fes = 0

    while temp > final_temp and iteration < max_iter and fes < maxFES:
        neighbor_solution = generate_neighbor_solution_mckp(current_solution, num_classes, items_per_class)
        neighbor_value, neighbor_weight = mckp_value(weights, prices, neighbor_solution)
        fes += 1

        if neighbor_weight <= capacity and (neighbor_value > current_value or random.uniform(0, 1) < np.exp((neighbor_value - current_value) / temp)):
            current_solution = neighbor_solution
            current_value = neighbor_value
            current_weight = neighbor_weight

            if current_value > best_value:
                best_solution = current_solution.copy()
                best_value = current_value
                best_weight = current_weight

        values.append(best_value)
        temp *= cooling_rate
        iteration += 1

    return best_solution, best_value, values, best_weight, fes



