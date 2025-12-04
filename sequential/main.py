import numpy as np


def schwefel(x):
    x = np.asarray(x)
    n = x.size
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def rosenbrock(x):

    x = np.asarray(x)
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def inicjalizuj_roj(objective, n_dim, bounds, swarm_size, random_state=None):
    low, high = bounds

    if random_state is not None:
        np.random.seed(random_state)

    positions = np.random.uniform(
        low=low, high=high, size=(swarm_size, n_dim)
    )
    velocities = np.zeros_like(positions)

    pbest_positions = positions.copy()
    pbest_values = np.apply_along_axis(objective, 1, positions)

    best_idx = np.argmin(pbest_values)
    gbest_position = pbest_positions[best_idx].copy()
    gbest_value = pbest_values[best_idx]

    best_history = [gbest_value]

    return (positions, velocities,
            pbest_positions, pbest_values,
            gbest_position, gbest_value,
            best_history)



def wykonaj_iteracje(objective,
                     positions, velocities,
                     pbest_positions, pbest_values,
                     gbest_position, gbest_value,
                     best_history,
                     bounds, w, c1, c2):

    low, high = bounds
    swarm_size, n_dim = positions.shape

    r1 = np.random.rand(swarm_size, n_dim)
    r2 = np.random.rand(swarm_size, n_dim)

    cognitive = c1 * r1 * (pbest_positions - positions)
    social = c2 * r2 * (gbest_position - positions)

    velocities = w * velocities + cognitive + social
    positions = positions + velocities

    positions = np.clip(positions, low, high)

    values = np.apply_along_axis(objective, 1, positions)

    improved = values < pbest_values
    pbest_values[improved] = values[improved]
    pbest_positions[improved] = positions[improved]

    best_idx = np.argmin(pbest_values)
    if pbest_values[best_idx] < gbest_value:
        gbest_value = pbest_values[best_idx]
        gbest_position = pbest_positions[best_idx].copy()

    best_history.append(gbest_value)

    return (positions, velocities,
            pbest_positions, pbest_values,
            gbest_position, gbest_value,
            best_history)



def stop_znane_optimum(gbest_position, x_opt, eps_opt):

    diff = gbest_position - x_opt
    norm = np.linalg.norm(diff)
    return norm <= eps_opt


def stop_brak_poprawy(best_history, m_no_improve, eps_no_improve):

    m = m_no_improve
    if len(best_history) <= m:
        return False

    f_old = best_history[-m - 1]
    f_new = best_history[-1]
    return abs(f_old - f_new) <= eps_no_improve



def uruchom_pso(objective,
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
                random_state=None):

    (positions, velocities,
     pbest_positions, pbest_values,
     gbest_position, gbest_value,
     best_history) = inicjalizuj_roj(
        objective, n_dim, bounds, swarm_size, random_state
    )

    for it in range(max_iters):
        (positions, velocities,
         pbest_positions, pbest_values,
         gbest_position, gbest_value,
         best_history) = wykonaj_iteracje(
            objective,
            positions, velocities,
            pbest_positions, pbest_values,
            gbest_position, gbest_value,
            best_history,
            bounds, w, c1, c2
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
    ns = [10, 50, 100]

    print("=== Zadanie 1: Schwefel ===")
    for n in ns:
        print(f"\nWymiar n = {n}")
        x_opt_sch = np.full(n, 420.9687)

        best_x, best_f, iters = uruchom_pso(
            objective=schwefel,
            n_dim=n,
            bounds=(-100.0, 500.0),
            swarm_size=60,
            max_iters=5000,
            tryb_stopu="known",
            x_opt=x_opt_sch,
            eps_opt=1e-3,
            random_state=0,
        )
        print("[Kryterium 1] iteracje:", iters)
        print("[Kryterium 1] najlepsze f(x):", best_f)

        best_x2, best_f2, iters2 = uruchom_pso(
            objective=schwefel,
            n_dim=n,
            bounds=(-100.0, 500.0),
            swarm_size=60,
            max_iters=5000,
            tryb_stopu="no_improve",
            m_no_improve=50,
            eps_no_improve=1e-6,
            random_state=0,
        )
        print("[Kryterium 2] iteracje:", iters2)
        print("[Kryterium 2] najlepsze f(x):", best_f2)

    print("\n\n=== Zadanie 2: Rosenbrock ===")
    for n in ns:
        print(f"\nWymiar n = {n}")
        x_opt_ros = np.ones(n)

        best_x, best_f, iters = uruchom_pso(
            objective=rosenbrock,
            n_dim=n,
            bounds=(-10.0, 10.0),
            swarm_size=60,
            max_iters=5000,
            tryb_stopu="known",
            x_opt=x_opt_ros,
            eps_opt=1e-3,
            random_state=0,
        )
        print("[Kryterium 1] iteracje:", iters)
        print("[Kryterium 1] najlepsze f(x):", best_f)

        best_x2, best_f2, iters2 = uruchom_pso(
            objective=rosenbrock,
            n_dim=n,
            bounds=(-10.0, 10.0),
            swarm_size=60,
            max_iters=5000,
            tryb_stopu="no_improve",
            m_no_improve=50,
            eps_no_improve=1e-8,
            random_state=0,
        )
        print("[Kryterium 2] iteracje:", iters2)
        print("[Kryterium 2] najlepsze f(x):", best_f2)


if __name__ == "__main__":
    eksperymenty()