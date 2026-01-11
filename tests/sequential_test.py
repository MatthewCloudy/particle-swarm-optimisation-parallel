import time
import pickle
import numpy as np
from benchmarks.benchmark_functions import rosenbrock, schwefel
from sequential.main import uruchom_pso
import statistics

# import os
# if not os.path.exists("./data"):
#     os.makedirs("./data")

ITERATIONS = 10

def eksperymenty():
    funkcje = [
        ("Schwefel", schwefel, (-100.0, 500.0)),
        ("Rosenbrock", rosenbrock, (-10.0, 10.0))
    ]

    ns = [2]

    for nazwa, objective, bounds in funkcje:
        print(f"\n=== Zadanie: {nazwa} ===")

        for n in ns:
            print(f"\nWymiar n = {n}")
            best_values = []
            iterations = []
            times = []
            times_div_iterations = []
            for iteration in range(ITERATIONS):

                # ustalamy optimum znane
                if nazwa == "Schwefel":
                    x_opt = np.full(n, 420.9687)
                else:
                    x_opt = np.ones(n)

                # ---------------------------------------------
                # Kryterium 1 — znane optimum
                # ---------------------------------------------
                start = time.time()
                best_x, best_f, iters, metadata, positions = uruchom_pso(
                    objective=objective,
                    n_dim=n,
                    bounds=bounds,
                    swarm_size=1000,
                    max_iters=10000,
                    tryb_stopu="known",
                    x_opt=x_opt,
                    eps_opt=1e-3,
                    random_state= 0
                )
                end = time.time()
                
                time_passed = end - start
                iterations.append(iters)
                best_values.append(best_f.item())
                times.append(time_passed)
                times_div_iterations.append(time_passed / iters)

            print(f"[Kryterium 1] iteracje (średnia, odchylenie): {statistics.mean(iterations)}, {statistics.stdev(iterations)}")
            print(f"[Kryterium 1] najlepsze f(x) (średnia, odchylenie): {statistics.mean(best_values)}, {statistics.stdev(best_values)}")
            print(f"[Kryterium 1] czas (średnia, odchylenie): {statistics.mean(times)}, {statistics.stdev(times)}")
            print(f"[Kryterium 1] czas (średnia, odchylenie): {statistics.mean(times_div_iterations)}, {statistics.stdev(times_div_iterations)}")
            # file = open(f'./data/sequential_{nazwa}_kryt_1_metadata.txt', 'wb')
            # pickle.dump(metadata, file)
            # file.close()
            #
            # file = open(f'./data/sequential_{nazwa}_kryt_1_points.txt', 'wb')
            # for position in positions:
            #     np.save(file, position)
            # file.close()

            # ---------------------------------------------
            # Kryterium 2 — brak poprawy
            # ---------------------------------------------
            for iteration in range(ITERATIONS):
                start = time.time()
                best_x2, best_f2, iters2, metadata, positions = uruchom_pso(
                    objective=objective,
                    n_dim=n,
                    bounds=bounds,
                    swarm_size=1000,
                    max_iters=10000,
                    tryb_stopu="no_improve",
                    m_no_improve=50,
                    eps_no_improve=1e-6,
                    random_state=0
                )
                end = time.time()

                time_passed = end - start
                iterations.append(iters)
                best_values.append(best_f.item())
                times.append(time_passed)
                times_div_iterations.append(time_passed / iters)

            print(f"[Kryterium 2] iteracje (średnia, odchylenie): {statistics.mean(iterations)}, {statistics.stdev(iterations)}")
            print(f"[Kryterium 2] najlepsze f(x) (średnia, odchylenie): {statistics.mean(best_values)}, {statistics.stdev(best_values)}")
            print(f"[Kryterium 2] czas (średnia, odchylenie): {statistics.mean(times)}, {statistics.stdev(times)}")
            print(f"[Kryterium 2] czas / iteracje (średnia, odchylenie): {statistics.mean(times_div_iterations)}, {statistics.stdev(times_div_iterations)}")

            # file = open(f'./data/sequential_{nazwa}_kryt_2_metadata.txt', 'wb')
            # pickle.dump(metadata, file)
            # file.close()
            #
            # file = open(f'./data/sequential_{nazwa}_kryt_2_points.txt', 'wb')
            # for position in positions:
            #     np.save(file, position)
            # file.close()


if __name__ == "__main__":
    eksperymenty()