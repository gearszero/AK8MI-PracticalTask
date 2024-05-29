import generate_items as gi
import time
import brute_force as bf
import simulated_annealing as sm
import numpy as np
import matplotlib.pyplot as plt


def main():
    num_items = 15
    num_classes = 8
    capacity = 100
    capacity_mkcp = 200

    prices_kp, weights_kp = gi.generate_knapsack_items(num_items)
    prices_mckp, weights_mckp = gi.generate_mckp_items(num_classes)

    brute_kp_fes = run_brute_force_kp(prices_kp, weights_kp, capacity)
    brute_mckp_fes = run_brute_force_mckp(prices_mckp, weights_mckp, num_classes, capacity_mkcp)
    run_simmulated_annealing_kp(prices_kp, weights_kp, brute_kp_fes, capacity)
    run_simmulated_annealing_mckp(prices_mckp, weights_mckp, brute_mckp_fes,num_classes, capacity_mkcp)


def run_brute_force_kp(prices, weights, capacity):
    print("\nCeny: ", prices)
    print("Vahy: ", weights)

    start_time = time.time()
    best_combination, best_value, maxFES, values = bf.brute_force_knapsack(prices, weights, capacity)
    end_time = time.time()

    print("Čas brute force KP:", end_time - start_time, "sekund")
    print("MaxFES brute force KP:", maxFES)
    print("Nejlepší kombinace předmětů:", best_combination)
    print("Maximální cena:", best_value)
    print("Celková hmotnost:", sum(weights[i] for i in best_combination))
    print("\n")
    return maxFES, best_value, values

def run_brute_force_mckp(prices, weights, num_classes, capacity):
    print("\nCeny: ", prices)
    print("Vahy: ", weights)

    start_time = time.time()
    best_combination, best_value, maxFES, values = bf.brute_force_mckp(prices, weights, capacity)
    end_time = time.time()

    print("Čas brute force MCKP:", end_time - start_time, "sekund")
    print("MaxFES brute force MCKP:", maxFES)
    print("Nejlepší kombinace předmětů [indexy]:", best_combination)
    print("Maximální cena:", best_value)
    if not best_combination:
        print("No solution found")
        return maxFES, best_value, values

    print("Celková hmotnost:", sum(weights[i][best_combination[i]] for i in range(num_classes)))

    print("\nPodrobnosti o vybraných předmětech:")
    for i in range(num_classes):
        item_index = best_combination[i]
        print(f"Třída {i + 1}: Předmět {item_index + 1}, Cena: {prices[i][item_index]}, Váha: {weights[i][item_index]}")
    print("\n")
    return maxFES, best_value, values


def run_simmulated_annealing_kp(prices, weights, maxFES_brute_force_kp, capacity):

    # Parametry simulovaného žíhání
    initial_temp = 100000
    final_temp = 0.01
    cooling_rate = 0.89
    max_iter = 50000

    start_time = time.time()
    best_solution, best_value, values, best_weight, FES_simulated_kp = sm.simulated_annealing_knapsack(weights, prices,
                                                                                                       capacity,
                                                                                                       initial_temp,
                                                                                                       final_temp,
                                                                                                       cooling_rate,
                                                                                                       max_iter,
                                                                                                       maxFES_brute_force_kp)
    end_time = time.time()

    print("\n")
    print("Čas simulovaného žíhání (KP):", end_time - start_time, "sekund")
    print("MaxFES pro SA:", FES_simulated_kp)


    print("Nejlepší nalezené řešení (indexy):", np.where(best_solution == 1)[0])
    print("Maximální cena:", best_value)
    print("Celková hmotnost:", best_weight)
    print("\n")

    # Konvergenční graf
    plt.plot(values)
    plt.xlabel('Iterace')
    plt.ylabel('Hodnota')
    plt.title('Konvergence simulovaného žíhání (KP)')
    plt.show()


def run_simmulated_annealing_mckp(prices, weights, maxFES_brute_force_mckp, num_classes, capacity):
    # Parametry simulovaného žíhání
    initial_temp = 100000
    final_temp = 0.01
    cooling_rate = 0.89
    max_iter = 50000

    # Spuštění simulovaného žíhání pro MCKP
    start_time = time.time()
    best_solution, best_value, values, best_weight, FES_simulated_mckp = sm.simulated_annealing_mckp(weights, prices,
                                                                                      capacity, initial_temp,
                                                                                      final_temp, cooling_rate,
                                                                                      max_iter, maxFES_brute_force_mckp)
    end_time = time.time()
    print("\n")
    print("Čas simulovaného žíhání (MCKP):", end_time - start_time, "sekund")
    print("FES simulovaného žíhání (MCKP):", FES_simulated_mckp)
    print("MaxFES brute force MCKP:", maxFES_brute_force_mckp)


    print("Nejlepší nalezené řešení (indexy):", best_solution)
    print("Maximální cena:", best_value)
    print("Celková hmotnost:", best_weight)

    # Konvergenční graf
    plt.plot(values)
    plt.xlabel('Iterace')
    plt.ylabel('Hodnota')
    plt.title('Konvergence simulovaného žíhání (MCKP)')
    plt.show()


def run_experiments_sa_kp(prices, weights, capacity, num_runs, initial_temp, final_temp, cooling_rate, max_iter, maxFES):
    all_runs_values = []
    best_value = -np.inf
    best_solution = None
    best_weight = 0
    best_time = 0
    best_FES = 0

    for _ in range(num_runs):
        start_time = time.time()
        solution, value, values, weight, FES = sm.simulated_annealing_knapsack(weights, prices, capacity, initial_temp, final_temp, cooling_rate, max_iter, maxFES)
        end_time = time.time()
        all_runs_values.append(values)

        if value > best_value:
            best_value = value
            best_solution = solution
            best_weight = weight
            best_time = end_time - start_time
            best_FES = FES

    print("\n")
    print("Čas simulovaného žíhání (KP):", best_time, "sekund")
    print("MaxFES pro SA:", best_FES)
    print("Nejlepší nalezené řešení (indexy):", np.where(best_solution == 1)[0])
    print("Maximální cena:", best_value)
    print("Celková hmotnost:", best_weight)
    print("\n")

    return all_runs_values


def run_experiments_sa_mckp(prices, weights, num_classes, capacity, num_runs, initial_temp, final_temp, cooling_rate, max_iter, maxFES):
    all_runs_values = []
    best_value = -np.inf
    best_solution = None
    best_weight = 0
    best_time = 0
    best_FES = 0

    for _ in range(num_runs):
        start_time = time.time()
        solution, value, values, weight, FES = sm.simulated_annealing_mckp(weights, prices, capacity, initial_temp, final_temp, cooling_rate, max_iter, maxFES)
        end_time = time.time()
        all_runs_values.append(values)

        if value > best_value:
            best_value = value
            best_solution = solution
            best_weight = weight
            best_time = end_time - start_time
            best_FES = FES

    print("\n")
    print("Čas simulovaného žíhání (MCKP):", best_time, "sekund")
    print("FES simulovaného žíhání (MCKP):", best_FES)
    print("Nejlepší nalezené řešení (indexy):", best_solution)
    print("Maximální cena:", best_value)
    print("Celková hmotnost:", best_weight)
    print("\n")

    return all_runs_values



def plot_sa_convergence(all_runs_values, title):
    plt.figure(figsize=(12, 8))  # Větší rozměry grafu
    for run_values in all_runs_values:
        plt.plot(run_values, alpha=0.5)
    plt.xlabel('Iterace')
    plt.ylabel('Hodnota')
    plt.title(title)
    plt.savefig(f'{title}.png', dpi=300)  # Vyšší DPI
    plt.show()

def plot_bf_convergence(values, title):
    plt.figure(figsize=(12, 8))  # Větší rozměry grafu
    plt.plot(values)
    plt.xlabel('Iterace')
    plt.ylabel('Hodnota')
    plt.title(title)
    plt.savefig(f'{title}.png', dpi=300)  # Vyšší DPI
    plt.show()

def plot_comparison(sa_values, bf_values, title):
    plt.figure(figsize=(12, 8))  # Větší rozměry grafu
    for run_values in sa_values:
        plt.plot(run_values, alpha=0.5, label='Simulated Annealing' if run_values == sa_values[0] else "")
    plt.plot(bf_values, color='r', linestyle='-', label='Brute Force')
    plt.xlabel('Iterace')
    plt.ylabel('Hodnota')
    plt.title(title)
    plt.legend()
    plt.savefig(f'{title}.png', dpi=300)  # Vyšší DPI
    plt.show()



def main_graphs():
    num_items = 15
    num_classes = 8
    capacity = 100
    capacity_mckp = 200
    num_runs = 30  # Počet opakování pro Simulated Annealing

    prices_kp, weights_kp = gi.generate_knapsack_items(num_items)
    prices_mckp, weights_mckp = gi.generate_mckp_items(num_classes)

    brute_kp_fes, best_value_kp, bf_values_kp = run_brute_force_kp(prices_kp, weights_kp, capacity)
    brute_mckp_fes, best_value_mckp, bf_values_mckp = run_brute_force_mckp(prices_mckp, weights_mckp, num_classes,
                                                                           capacity_mckp)

    # Spuštění Simulated Annealing pro KP
    sa_values_kp = run_experiments_sa_kp(prices_kp, weights_kp, capacity, num_runs, 100000, 0.01, 0.89, 50000,
                                         brute_kp_fes)
    # Spuštění Simulated Annealing pro MCKP
    sa_values_mckp = run_experiments_sa_mckp(prices_mckp, weights_mckp, num_classes, capacity_mckp, num_runs,
                                             100000, 0.01, 0.89, 50000, brute_mckp_fes)

    # Vykreslení konvergenčních grafů pro KP
    plot_sa_convergence(sa_values_kp, 'Konvergence simulovaného žíhání (KP)')
    plot_bf_convergence(bf_values_kp, 'Konvergence brute force (KP)')

    # Vykreslení konvergenčních grafů pro MCKP
    plot_sa_convergence(sa_values_mckp, 'Konvergence simulovaného žíhání (MCKP)')
    plot_bf_convergence(bf_values_mckp, 'Konvergence brute force (MCKP)')

    # Vykreslení porovnání grafů pro KP a MCKP
    plot_comparison(sa_values_kp, bf_values_kp, 'Porovnání SA a BF (KP)')
    plot_comparison(sa_values_mckp, bf_values_mckp, 'Porovnání SA a BF (MCKP)')


if __name__ == '__main__':
    # main()
    main_graphs()
