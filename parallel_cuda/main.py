import time
import pickle
import numpy as np
import cupy as cp

from cupyx.profiler import benchmark



def stop_znane_optimum(gbest_position, x_opt, eps_opt):

    diff = gbest_position - x_opt
    norm = cp.linalg.norm(diff)
    return norm <= eps_opt


def stop_brak_poprawy(best_history, m_no_improve, eps_no_improve):

    m = m_no_improve
    if len(best_history) <= m:
        return False

    f_old = best_history[-m - 1]
    f_new = best_history[-1]
    return cp.abs(f_old - f_new) <= eps_no_improve

calculate_velocities = cp.ElementwiseKernel(

    'float64 velocity, float64 position, float64 best_local, float64 best_global, '
    'float64 w, float64 c1, float64 c2, float64 r1, float64 r2', 

    'float64 new_velocity',

    'new_velocity = w * velocity + c1 * r1 * (best_local - position) '
    '+ c2 * r2 * (best_global - position)',

    'calculate_velocities')

def calculate_velocities2(velocity,position,best_local,best_global,w,c1,c2,r1,r2):
    return velocity + position

def inicjalizuj_roj(objective, n_dim, bounds, swarm_size, random_state=None):
    low, high = bounds

    if random_state is not None:
        cp.random.seed(random_state)

    positions = cp.random.uniform(low=low, high=high, size=(swarm_size, n_dim))
    velocities = cp.zeros_like(positions)

    pbest_positions = positions.copy()
    pbest_values = objective(positions)

    best_idx = cp.argmin(pbest_values)
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

    r1 = cp.random.rand(swarm_size, n_dim)
    r2 = cp.random.rand(swarm_size, n_dim)

    velocities = calculate_velocities(velocities, positions, pbest_positions, gbest_position, w, c1, c2, r1, r2)

    positions += velocities
    np.clip(positions, low, high, out=positions)

    values = objective(positions)

    improved_mask = values < pbest_values
    pbest_values[improved_mask] = values[improved_mask]
    pbest_positions[improved_mask] = positions[improved_mask]

    current_best_idx = np.argmin(pbest_values)
    current_best_val = pbest_values[current_best_idx]

    if current_best_val < gbest_value:
        gbest_value = current_best_val
        gbest_position[:] = pbest_positions[current_best_idx]

    return (positions, velocities,
            pbest_positions, pbest_values,
            gbest_position, gbest_value)


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
        positions_list.append(cp.asnumpy(positions))
        (
            positions,
            velocities,
            pbest_positions,
            pbest_values,
            gbest_position,
            gbest_value
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
    
    return cp.asnumpy(gbest_position), cp.asnumpy(gbest_value), liczba_iteracji, metadata, positions_list

if __name__ == "__main__":
    def my_func(a):

        return cp.sqrt(cp.sum(a**2, axis=-1))


    a = cp.random.random((256, 1024))

    print(benchmark(my_func, (a,), n_repeat=20))

    NUM = 1000
    DIM = 10000
    velocity = cp.random.random((NUM, DIM), dtype=np.float64)
    position = cp.random.random((NUM, DIM), dtype=np.float64)
    best_local = cp.random.random((NUM, DIM), dtype=np.float64)
    best_global = cp.random.random((1, DIM), dtype=np.float64)
    w = cp.array([0.5], dtype=np.float64)
    c1 = cp.array([0.25], dtype=np.float64)
    c2 = cp.array([0.75], dtype=np.float64)
    r1 = cp.random.random((NUM, DIM), dtype=np.float64)
    r2 = cp.random.random((NUM, DIM), dtype=np.float64)



    print(benchmark(calculate_velocities, (velocity,position,best_local,best_global,w,c1,c2,r1,r2), n_repeat=20))