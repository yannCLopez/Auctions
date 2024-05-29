import numpy as np
from joblib import Parallel, delayed, cpu_count

# Function to perform the draw and calculation
def draw_and_calculate():
    #Chose distribution from which values are drawn:
    #v1_A, v2_A, v1_beta, v2_beta = np.random.normal(0, 1, 4)
    #v1_A, v2_A, v1_beta, v2_beta = np.random.poisson(1, 4)
    #v1_A, v2_A, v1_beta, v2_beta = np.random.gamma(1, 1, 4)
    #v1_A, v2_A, v1_beta, v2_beta = np.random.beta(1, 1, 4)
    #v1_A, v2_A, v1_beta, v2_beta = np.random.chisquare(1, 4)
    #v1_A, v2_A, v1_beta, v2_beta = np.random.uniform(0, 1, 4)
    v1_A, v2_A, v1_beta, v2_beta = np.random.exponential(1, 4)

    if v1_beta + v2_beta >= v1_A + v2_A:
        E1 = v1_beta + v2_beta - v2_A
        E2 = v1_beta + v2_beta - v1_A
        return E1, E2
    return None

# Main function to run the simulation
def run_simulation(n_draws):
    num_cores = cpu_count() - 1
    results = Parallel(n_jobs=num_cores)(delayed(draw_and_calculate)() for _ in range(n_draws))
    list_1 = [result[0] for result in results if result is not None]
    list_2 = [result[1] for result in results if result is not None]
    avg_list_1 = np.mean(list_1) if list_1 else None
    avg_list_2 = np.mean(list_2) if list_2 else None
    return avg_list_1, avg_list_2

# Test the function with 10 draws
test_avg_list_1, test_avg_list_2 = run_simulation(10000)
print(test_avg_list_1, test_avg_list_2)
