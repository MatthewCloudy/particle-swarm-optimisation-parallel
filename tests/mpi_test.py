from mpi4py import MPI
import time
import numpy as np
from benchmarks.benchmark_functions import rosenbrock, schwefel
from parallel_mpi.main import uruchom_pso


def eksperymenty():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    funkcje = [
        ("Schwefel", schwefel, (-100.0, 500.0)),
        ("Rosenbrock", rosenbrock, (-10.0, 10.0))
    ]

    ns = [10]

    for nazwa, objective, bounds in funkcje:
        if rank == 0:
            print(f"\n=== Zadanie: {nazwa} ===")

        for n in ns:
            if rank == 0:
                print(f"\nWymiar n = {n}")

            if nazwa == "Schwefel":
                x_opt = np.full(n, 420.9687)
            else:
                x_opt = np.ones(n)

            # ---------------------------------------------
            # Kryterium 1 — znane optimum
            # ---------------------------------------------
            comm.Barrier()
            start = time.time()

            wynik = uruchom_pso(
                objective=objective,
                n_dim=n,
                bounds=bounds,
                comm=comm,
                swarm_size=1000,
                max_iters=10000,
                tryb_stopu="known",
                x_opt=x_opt,
                eps_opt=1e-3,
                random_state=0
            )

            comm.Barrier()
            end = time.time()

            if rank == 0:
                best_x, best_f, iters, metadata, positions = wynik
                print("[Kryterium 1] iteracje:", iters)
                print("[Kryterium 1] najlepsze f(x):", best_f)
                print("[Kryterium 1] czas:", end - start)

            # ---------------------------------------------
            # Kryterium 2 — brak poprawy
            # ---------------------------------------------
            comm.Barrier()
            start = time.time()

            wynik2 = uruchom_pso(
                objective=objective,
                n_dim=n,
                bounds=bounds,
                comm=comm,
                swarm_size=1000,
                max_iters=10000,
                tryb_stopu="no_improve",
                m_no_improve=50,
                eps_no_improve=1e-6,
                random_state=0
            )

            comm.Barrier()
            end = time.time()

            if rank == 0:
                best_x2, best_f2, iters2, metadata, positions = wynik2
                print("[Kryterium 2] iteracje:", iters2)
                print("[Kryterium 2] najlepsze f(x):", best_f2)
                print("[Kryterium 2] czas:", end - start)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        print("Liczba procesów MPI:", comm.Get_size())
    eksperymenty()