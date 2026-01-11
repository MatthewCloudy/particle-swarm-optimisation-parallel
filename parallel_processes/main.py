from collections import deque

import numpy as np
from benchmarks.benchmark_functions import rosenbrock, schwefel
from benchmarks.stop_conditions import stop_brak_poprawy_deque, stop_znane_optimum
from multiprocessing import Pool, cpu_count
import time
import pickle

def eval_block(block, objective_func):
    return objective_func(block)

def chunkify(arr, n_chunks):
    if n_chunks > len(arr):
        n_chunks = len(arr)
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
    pbest_values = objective(positions)

    best_idx = np.argmin(pbest_values)
    gbest_position = pbest_positions[best_idx].copy()
    gbest_value = pbest_values[best_idx]

    return (positions, velocities,
            pbest_positions, pbest_values,
            gbest_position, gbest_value)


def wykonaj_iteracje(
    pool,
    objective,
    positions,
    velocities,
    pbest_positions,
    pbest_values,
    gbest_position,
    gbest_value,
    bounds,
    w,
    c1,
    c2,
):

    low, high = bounds
    swarm_size, n_dim = positions.shape

    r1 = np.random.rand(swarm_size, n_dim)
    r2 = np.random.rand(swarm_size, n_dim)

    r1 *= c1
    r1 *= (pbest_positions - positions)

    r2 *= c2
    r2 *= (gbest_position - positions)

    velocities *= w
    velocities += r1
    velocities += r2

    positions += velocities
    np.clip(positions, low, high, out=positions)

    nprocs = pool._processes
    chunks = chunkify(positions, nprocs)
    args = [(chunk, objective) for chunk in chunks]

    block_results = pool.starmap(eval_block, args)
    values = np.concatenate(block_results)

    improved = values < pbest_values
    pbest_values[improved] = values[improved]
    pbest_positions[improved] = positions[improved]

    current_best_idx = np.argmin(pbest_values)
    current_best_val = pbest_values[current_best_idx]
    if current_best_val < gbest_value:
        gbest_value = current_best_val
        gbest_position[:] = pbest_positions[current_best_idx]

    return (positions, velocities,
            pbest_positions, pbest_values,
            gbest_position, gbest_value)


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
    n_cores = cpu_count()
    positions_list = []
    (
        positions,
        velocities,
        pbest_positions,
        pbest_values,
        gbest_position,
        gbest_value,
    ) = inicjalizuj_roj(objective, n_dim, bounds, swarm_size, random_state)

    convergence_window = deque(maxlen=m_no_improve + 1)
    convergence_window.append(float(gbest_value))
    iterations_done = 0
    zapisuj_pozycje = (n_dim == 2)
    for it in range(max_iters):
        if zapisuj_pozycje:
            positions_list.append(positions.copy())

        positions_list.append(positions.copy())

        (positions, velocities,
         pbest_positions, pbest_values,
         gbest_position, gbest_value) = wykonaj_iteracje(
            pool,
            objective,
            positions, velocities,
            pbest_positions, pbest_values,
            gbest_position, gbest_value,
            bounds, w, c1, c2
        )

        convergence_window.append(float(gbest_value))
        iterations_done = it + 1

        if tryb_stopu == "known" and x_opt is not None:
            if stop_znane_optimum(gbest_position, x_opt, eps_opt):
                break
        elif tryb_stopu == "no_improve":
            if stop_brak_poprawy_deque(convergence_window, eps_no_improve):
                break

    metadata = {
        "Iteracje": iterations_done,
        "Liczba punktów": swarm_size,
        "Funkcja": objective.__name__,
        "Ograniczenia": bounds,
        "Procesory": n_cores
    }

    return gbest_position, gbest_value, iterations_done, metadata, positions_list

