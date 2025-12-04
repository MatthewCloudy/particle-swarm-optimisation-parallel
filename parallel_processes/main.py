import numpy as np
from benchmarks.benchmark_functions import rosenbrock, schwefel
from benchmarks.stop_conditions import stop_brak_poprawy, stop_znane_optimum
from multiprocessing import Pool, cpu_count
import time

def eval_block(block, objective):
    return [objective(x) for x in block]


def chunkify(arr, n_chunks):
    chunk_size = len(arr) // n_chunks
    chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_chunks - 1 else len(arr)
        chunks.append(arr[start:end])
    return chunks

def inicjalizuj_roj(objective, n_dim, bounds, swarm_size, random_state=None):
    low, high = bounds

    if random_state is not None:
        np.random.seed(random_state)

    positions = np.random.uniform(low=low, high=high, size=(swarm_size, n_dim))
    velocities = np.zeros_like(positions)

    pbest_positions = positions.copy()
    pbest_values = np.apply_along_axis(objective, 1, positions)

    best_idx = np.argmin(pbest_values)
    gbest_position = pbest_positions[best_idx].copy()
    gbest_value = pbest_values[best_idx]

    best_history = [gbest_value]

    return (
        positions,
        velocities,
        pbest_positions,
        pbest_values,
        gbest_position,
        gbest_value,
        best_history,
    )


def wykonaj_iteracje(
    pool,
    objective,
    positions,
    velocities,
    pbest_positions,
    pbest_values,
    gbest_position,
    gbest_value,
    best_history,
    bounds,
    w,
    c1,
    c2,
):

    low, high = bounds
    swarm_size, n_dim = positions.shape

    r1 = np.random.rand(swarm_size, n_dim)
    r2 = np.random.rand(swarm_size, n_dim)

    cognitive = c1 * r1 * (pbest_positions - positions)
    social = c2 * r2 * (gbest_position - positions)

    velocities = w * velocities + cognitive + social
    positions = positions + velocities

    positions = np.clip(positions, low, high)

    nprocs = pool._processes
    chunks = chunkify(positions, nprocs)
    args = [(chunk, objective) for chunk in chunks]

    block_results = pool.starmap(eval_block, args)
    values = np.concatenate(block_results)

    improved = values < pbest_values
    pbest_values[improved] = values[improved]
    pbest_positions[improved] = positions[improved]

    best_idx = np.argmin(pbest_values)
    if pbest_values[best_idx] < gbest_value:
        gbest_value = pbest_values[best_idx]
        gbest_position = pbest_positions[best_idx].copy()

    best_history.append(gbest_value)

    return (
        positions,
        velocities,
        pbest_positions,
        pbest_values,
        gbest_position,
        gbest_value,
        best_history,
    )


def uruchom_pso(
    pool,
    objective,
    n_dim,
    bounds,
    swarm_size=50,
    w=0.7,
    c1=1.5,
    c2=1.5,
    max_iters=2000,
    tryb_stopu="known",
    x_opt=None,
    eps_opt=1e-6,
    m_no_improve=50,
    eps_no_improve=1e-6,
    random_state=None,
):

    (
        positions,
        velocities,
        pbest_positions,
        pbest_values,
        gbest_position,
        gbest_value,
        best_history,
    ) = inicjalizuj_roj(objective, n_dim, bounds, swarm_size, random_state)

    for it in range(max_iters):
        (
            positions,
            velocities,
            pbest_positions,
            pbest_values,
            gbest_position,
            gbest_value,
            best_history,
        ) = wykonaj_iteracje(
            pool,
            objective,
            positions,
            velocities,
            pbest_positions,
            pbest_values,
            gbest_position,
            gbest_value,
            best_history,
            bounds,
            w,
            c1,
            c2,
        )

        if tryb_stopu == "known" and x_opt is not None:
            if stop_znane_optimum(gbest_position, x_opt, eps_opt):
                break
        elif tryb_stopu == "no_improve":
            if stop_brak_poprawy(best_history, m_no_improve, eps_no_improve):
                break

    liczba_iteracji = len(best_history) - 1
    return gbest_position, gbest_value, liczba_iteracji

def eksperymenty():
    funkcje = [
        ("Schwefel", schwefel, (-100.0, 500.0)),
        ("Rosenbrock", rosenbrock, (-10.0, 10.0))
    ]

    ns = [10, 50, 100]

    for nazwa, objective, bounds in funkcje:
        print(f"\n=== Zadanie: {nazwa} ===")

        for n in ns:
            print(f"\nWymiar n = {n}")

            # ustalamy optimum znane
            if nazwa == "Schwefel":
                x_opt = np.full(n, 420.9687)
            else:
                x_opt = np.ones(n)

            # ---------------------------------------------
            # Kryterium 1 — znane optimum
            # ---------------------------------------------
            with Pool(processes=cpu_count()) as pool:
                start = time.time()
                best_x, best_f, iters = uruchom_pso(
                    objective=objective,
                    n_dim=n,
                    bounds=bounds,
                    pool=pool,
                    swarm_size=10000,
                    max_iters=5000,
                    tryb_stopu="known",
                    x_opt=x_opt,
                    eps_opt=1e-3,
                    random_state=0
                )
                end = time.time()

            print("[Kryterium 1] iteracje:", iters)
            print("[Kryterium 1] najlepsze f(x):", best_f)
            print("[Kryterium 1] czas:", end - start)

            # ---------------------------------------------
            # Kryterium 2 — brak poprawy
            # ---------------------------------------------
            with Pool(processes=cpu_count()) as pool:
                start = time.time()
                best_x2, best_f2, iters2 = uruchom_pso(
                    objective=objective,
                    n_dim=n,
                    bounds=bounds,
                    pool=pool,
                    swarm_size=10000,
                    max_iters=5000,
                    tryb_stopu="no_improve",
                    m_no_improve=50,
                    eps_no_improve=1e-6,
                    random_state=0
                )
                end = time.time()

            print("[Kryterium 2] iteracje:", iters2)
            print("[Kryterium 2] najlepsze f(x):", best_f2)
            print("[Kryterium 2] czas:", end - start)


if __name__ == "__main__":
    print("Liczba procesów:", cpu_count())
    eksperymenty()