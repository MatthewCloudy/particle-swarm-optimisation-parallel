from collections import deque
import numpy as np
from mpi4py import MPI
from benchmarks.stop_conditions import stop_brak_poprawy_deque, stop_znane_optimum

def get_local_indices(total_size, size, rank):
    base = total_size // size
    rem = total_size % size

    counts = [base + 1 if i < rem else base for i in range(size)]
    starts = [sum(counts[:i]) for i in range(size)]

    return starts[rank], starts[rank] + counts[rank]


def inicjalizuj_roj(objective, n_dim, bounds, swarm_size, comm, random_state=None):
    rank = comm.Get_rank()
    size = comm.Get_size()
    low, high = bounds

    if random_state is not None:
        np.random.seed(random_state)

    all_positions = np.random.uniform(low=low, high=high, size=(swarm_size, n_dim))

    my_start, my_end = get_local_indices(swarm_size, size, rank)

    positions = all_positions[my_start:my_end].copy()
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
        total_swarm_size,
        my_indices
):
    low, high = bounds
    _, n_dim = positions.shape
    my_start, my_end = my_indices

    r1_all = np.random.rand(total_swarm_size, n_dim)
    r2_all = np.random.rand(total_swarm_size, n_dim)

    r1 = r1_all[my_start:my_end]
    r2 = r2_all[my_start:my_end]

    r1 *= c1
    r1 *= (pbest_positions - positions)

    r2 *= c2
    r2 *= (gbest_position - positions)

    velocities *= w
    velocities += r1
    velocities += r2

    positions += velocities
    np.clip(positions, low, high, out=positions)

    values = objective(positions)

    improved = values < pbest_values
    pbest_values[improved] = values[improved]
    pbest_positions[improved] = positions[improved]

    current_local_best_idx = np.argmin(pbest_values)
    current_local_best_val = pbest_values[current_local_best_idx]
    current_local_best_pos = pbest_positions[current_local_best_idx]

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

    my_indices = get_local_indices(swarm_size, size, rank)

    positions_list = []

    (
        positions,
        velocities,
        pbest_positions,
        pbest_values,
        local_best_pos,
        local_best_val
    ) = inicjalizuj_roj(objective, n_dim, bounds, swarm_size, comm, random_state)

    all_init_bests = comm.allgather((local_best_val, local_best_pos))
    gbest_value, gbest_position = min(all_init_bests, key=lambda x: x[0])

    convergence_window = deque(maxlen=m_no_improve + 1)
    convergence_window.append(float(gbest_value))
    iterations_done = 0
    zapisuj_pozycje = (n_dim == 2)

    for it in range(max_iters):
        if zapisuj_pozycje:
            positions_list.append(positions.copy())

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
            swarm_size,
            my_indices
        )

        if current_gbest_value < gbest_value:
            gbest_value = current_gbest_value

        convergence_window.append(float(gbest_value))
        iterations_done = it + 1

        if tryb_stopu == "known" and x_opt is not None:
            if stop_znane_optimum(gbest_position, x_opt, eps_opt):
                break
        elif tryb_stopu == "no_improve":
            if stop_brak_poprawy_deque(convergence_window, eps_no_improve):
                break

    liczba_iteracji = iterations_done
    metadata = {"Iteracje": liczba_iteracji, "Liczba punktów": swarm_size,
                "Funkcja": objective.__name__, "Ograniczenia": bounds}

    return gbest_position, gbest_value, liczba_iteracji, metadata, positions_list