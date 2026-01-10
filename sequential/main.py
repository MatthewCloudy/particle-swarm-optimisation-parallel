import numpy as np
from benchmarks.benchmark_functions import rosenbrock, schwefel
from benchmarks.stop_conditions import stop_brak_poprawy, stop_znane_optimum
import time
import pickle

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

    values = objective(positions)

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
    positions_list = []
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
        positions_list.append(positions)
        (
            positions,
            velocities,
            pbest_positions,
            pbest_values,
            gbest_position,
            gbest_value,
            best_history,
        ) = wykonaj_iteracje(
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
    metadata = {"Iteracje": liczba_iteracji, "Liczba punktów": swarm_size, 
                "Funkcja": objective.__name__, "Ograniczenia" : bounds}
    
    return gbest_position, gbest_value, liczba_iteracji, metadata, positions_list



