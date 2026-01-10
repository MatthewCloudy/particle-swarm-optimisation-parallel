import numpy as np
from mpi4py import MPI
from benchmarks.stop_conditions import stop_brak_poprawy, stop_znane_optimum

def inicjalizuj_roj(objective, n_dim, bounds, local_swarm_size, rank, random_state=None):
    low, high = bounds

    #To jest po to żeby różne instancje miały różne pozycje początkowe gdy random_state jest ustawiony
    if random_state is not None:
        np.random.seed(random_state + rank)


    positions = np.random.uniform(low=low, high=high, size=(local_swarm_size, n_dim))
    velocities = np.zeros_like(positions)

    pbest_positions = positions.copy()
    pbest_values = objective(positions)

    local_best_idx = np.argmin(pbest_values)
    local_best_pos = pbest_positions[local_best_idx].copy()
    local_best_val = pbest_values[local_best_idx]

    return (
        positions,
        velocities,
        pbest_positions,
        pbest_values,
        local_best_pos,
        local_best_val
    )


def wykonaj_iteracje(
        comm,
        objective,
        positions,
        velocities,
        pbest_positions,
        pbest_values,
        gbest_position,
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

    current_local_best_idx = np.argmin(pbest_values)
    current_local_best_val = pbest_values[current_local_best_idx]
    current_local_best_pos = pbest_positions[current_local_best_idx]

    # Zbieranie najlepszych wyników z wszystkich instancji
    all_bests = comm.allgather((current_local_best_val, current_local_best_pos))

    best_val_global, best_pos_global = min(all_bests, key=lambda x: x[0])

    return (
        positions,
        velocities,
        pbest_positions,
        pbest_values,
        best_pos_global,
        best_val_global
    )


def uruchom_pso(
        objective,
        n_dim,
        bounds,
        comm,
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
    rank = comm.Get_rank()
    size = comm.Get_size()

    base_count = swarm_size // size
    remainder = swarm_size % size
    local_swarm_size = base_count + 1 if rank < remainder else base_count

    positions_list = []

    (
        positions,
        velocities,
        pbest_positions,
        pbest_values,
        local_best_pos,
        local_best_val
    ) = inicjalizuj_roj(objective, n_dim, bounds, local_swarm_size, rank, random_state)

    all_init_bests = comm.allgather((local_best_val, local_best_pos))
    gbest_value, gbest_position = min(all_init_bests, key=lambda x: x[0])

    best_history = [gbest_value]

    for it in range(max_iters):
        positions_list.append(positions)

        (
            positions,
            velocities,
            pbest_positions,
            pbest_values,
            gbest_position,
            current_gbest_value
        ) = wykonaj_iteracje(
            comm,
            objective,
            positions,
            velocities,
            pbest_positions,
            pbest_values,
            gbest_position,
            bounds,
            w,
            c1,
            c2,
        )

        if current_gbest_value < gbest_value:
            gbest_value = current_gbest_value

        best_history.append(gbest_value)

        if tryb_stopu == "known" and x_opt is not None:
            if stop_znane_optimum(gbest_position, x_opt, eps_opt):
                break
        elif tryb_stopu == "no_improve":
            if stop_brak_poprawy(best_history, m_no_improve, eps_no_improve):
                break

    liczba_iteracji = len(best_history) - 1
    metadata = {"Iteracje": liczba_iteracji, "Liczba punktów": swarm_size,
                "Funkcja": objective.__name__, "Ograniczenia": bounds}

    return gbest_position, gbest_value, liczba_iteracji, metadata, positions_list