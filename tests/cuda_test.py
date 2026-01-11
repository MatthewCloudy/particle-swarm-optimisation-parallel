from benchmarks.benchmark_functions import rosenbrock, schwefel
import cupy as cp
import time
from parallel_cuda.main import uruchom_pso
import pickle
import numpy as np
import os

if not os.path.exists("./data"):
    os.makedirs("./data")

def eksperymenty():
    funkcje = [
        ("Schwefel", schwefel, (-100.0, 500.0)),
        ("Rosenbrock", rosenbrock, (-10.0, 10.0))
    ]

    ns = [10]

    for nazwa, objective, bounds in funkcje:
        print(f"\n=== Zadanie: {nazwa} ===")

        for n in ns:
            print(f"\nWymiar n = {n}")

            # ustalamy optimum znane
            if nazwa == "Schwefel":
                x_opt = cp.full(n, 420.9687)
            else:
                x_opt = cp.ones(n)

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
                random_state=0
            )
            end = time.time()

            print("[Kryterium 1] iteracje:", iters)
            print("[Kryterium 1] najlepsze f(x):", best_f)
            print("[Kryterium 1] czas:", end - start)

            file = open(f'./data/parallel_{nazwa}_kryt_1_metadata.txt', 'wb')
            pickle.dump(metadata, file)
            file.close()
            
            file = open(f'./data/parallel_{nazwa}_kryt_1_points.txt', 'wb')
            for position in positions:
                np.save(file, position)
            file.close()

            # ---------------------------------------------
            # Kryterium 2 — brak poprawy
            # ---------------------------------------------
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

            print("[Kryterium 2] iteracje:", iters2)
            print("[Kryterium 2] najlepsze f(x):", best_f2)
            print("[Kryterium 2] czas:", end - start)

            file = open(f'./data/parallel_{nazwa}_kryt_2_metadata.txt', 'wb')
            print(metadata)
            pickle.dump(metadata, file)
            file.close()
            
            file = open(f'./data/parallel_{nazwa}_kryt_2_points.txt', 'wb')
            for position in positions:
                np.save(file, position)
            file.close()


if __name__ == "__main__":
    eksperymenty()