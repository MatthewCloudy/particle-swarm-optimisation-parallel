import sysconfig
import threading
import time
import numpy as np
import statistics
from collections import deque
import os
import sys

def sprawdz_srodowisko():
    print("\n" + "=" * 50)
    print("     DIAGNOSTYKA ŚRODOWISKA PYTHONA (GIL & THREADS)")
    print("=" * 50)

    # 1. Wersja Pythona
    print(f"Wersja Pythona: {sys.version.split()[0]}")

    # 2. Sprawdzenie flagi kompilacji (sysconfig)
    gil_config = sysconfig.get_config_var('Py_GIL_DISABLED')
    print(f"Build Configuration (Py_GIL_DISABLED): {gil_config} (1 = Free-threaded build)")

    # 3. Sprawdzenie runtime (tylko dla Python 3.13+)
    gil_status = "Nieznany (Stary Python)"
    try:
        if hasattr(sys, '_is_gil_enabled'):
            if not sys._is_gil_enabled():
                gil_status = "🚀 WYŁĄCZONY (Free-Threading AKTYWNY!)"
            else:
                gil_status = "🐢 WŁĄCZONY (Standardowy tryb)"
        else:
            gil_status = "🐢 WŁĄCZONY (Brak funkcji _is_gil_enabled)"
    except Exception as e:
        gil_status = f"Błąd sprawdzania: {e}"

    print(f"Status GIL w tym momencie: {gil_status}")

    # 4. Liczba dostępnych rdzeni
    cpu_count = os.cpu_count()
    print(f"Dostępne rdzenie logiczne CPU: {cpu_count}")
    print("=" * 50 + "\n")

def schwefel(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    n = x.shape[1]
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)


def rosenbrock(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    return np.sum(
        100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1.0 - x[:, :-1]) ** 2,
        axis=1,
    )


def stop_znane_optimum(gbest_position, x_opt, eps_opt):
    diff = gbest_position - x_opt
    norm = np.linalg.norm(diff)
    return norm <= eps_opt


def stop_brak_poprawy_deque(history_deque, eps_no_improve):
    if len(history_deque) < history_deque.maxlen:
        return False
    return abs(history_deque[0] - history_deque[-1]) <= eps_no_improve


def get_chunks(swarm_size, n_threads):
    base, rest = divmod(swarm_size, n_threads)
    ranges = []
    start = 0
    for i in range(n_threads):
        extra = 1 if i < rest else 0
        end = start + base + extra
        ranges.append((start, end))
        start = end
    return ranges


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
    return (
        positions,
        velocities,
        pbest_positions,
        pbest_values,
        gbest_position,
        gbest_value,
    )


# ==========================================
# 2. WORKER WĄTKU (zoptymalizowany pod RAM / alokacje)
# ==========================================

def pso_worker(
    t_id,
    range_start,
    range_end,
    objective,
    positions,
    velocities,
    pbest_positions,
    pbest_values,
    shared_gbest_container,
    bounds,
    w,
    c1,
    c2,
    barrier_start,
    barrier_end,
    stop_event,
):
    low, high = bounds

    # --- Slicing robimy tylko raz (views na pamięć wspólną) ---
    pos_slice = positions[range_start:range_end]
    vel_slice = velocities[range_start:range_end]
    pbest_pos_slice = pbest_positions[range_start:range_end]
    pbest_val_slice = pbest_values[range_start:range_end]

    slice_size, n_dim = pos_slice.shape

    # --- Bufory prealokowane, żeby nie robić nowych tablic co iterację ---
    r1 = np.empty_like(pos_slice)
    r2 = np.empty_like(pos_slice)
    tmp = np.empty_like(pos_slice)

    while True:
        try:
            # Czekamy na sygnał startu iteracji
            barrier_start.wait(timeout=10.0)
        except threading.BrokenBarrierError:
            break

        if stop_event.is_set():
            break

        # --- Losowe liczby (globalny RNG) do prealokowanych macierzy ---
        r1[:] = np.random.rand(slice_size, n_dim)
        r2[:] = np.random.rand(slice_size, n_dim)

        # Aktualny globalny gbest (współdzielony)
        gbest_pos_curr = shared_gbest_container[0]

        # ============================
        # AKTUALIZACJA PRĘDKOŚCI
        # ============================

        # vel = w * vel
        vel_slice *= w

        # Część kognitywna: c1 * r1 * (pbest - pos)
        # tmp = (pbest - pos)
        np.subtract(pbest_pos_slice, pos_slice, out=tmp)
        tmp *= r1
        tmp *= c1
        vel_slice += tmp

        # Część społeczna: c2 * r2 * (gbest - pos)
        # tmp = (gbest - pos)  (broadcastowanie gbest na wszystkie wiersze)
        np.subtract(gbest_pos_curr, pos_slice, out=tmp)
        tmp *= r2
        tmp *= c2
        vel_slice += tmp

        # ============================
        # AKTUALIZACJA POZYCJI
        # ============================
        pos_slice += vel_slice
        np.clip(pos_slice, low, high, out=pos_slice)

        # ============================
        # OCENA FUNKCJI CELU
        # ============================
        curr_vals = objective(pos_slice)

        # ============================
        # AKTUALIZACJA LOKALNYCH PBEST
        # ============================
        improved = curr_vals < pbest_val_slice
        if np.any(improved):
            pbest_val_slice[improved] = curr_vals[improved]
            pbest_pos_slice[improved] = pos_slice[improved]

        # Zgłaszamy koniec pracy w tej iteracji
        try:
            barrier_end.wait(timeout=10.0)
        except threading.BrokenBarrierError:
            break


# ==========================================
# 3. GŁÓWNA FUNKCJA STERUJĄCA
# ==========================================

def uruchom_pso_threaded(
    objective,
    n_dim,
    bounds,
    swarm_size,
    max_iters,
    tryb_stopu,
    x_opt,
    eps_opt,
    m_no_improve,
    eps_no_improve,
    random_state,
    n_threads=None,
    w=0.7,
    c1=1.5,
    c2=1.5,
):
    if n_threads is None:
        n_threads = os.cpu_count() or 4
    print(f"   -> [SYSTEM] Uruchamiam obliczenia na {n_threads} wątkach.")
    (
        positions,
        velocities,
        pbest_positions,
        pbest_values,
        gbest_position,
        gbest_value,
    ) = inicjalizuj_roj(objective, n_dim, bounds, swarm_size, random_state)

    # shared_gbest_container = [gbest_position, gbest_value]
    shared_gbest = [gbest_position, gbest_value]

    ranges = get_chunks(swarm_size, n_threads)

    barrier_start = threading.Barrier(n_threads + 1)
    barrier_end = threading.Barrier(n_threads + 1)
    stop_event = threading.Event()
    threads = []

    # Start wątków
    for t_id in range(n_threads):
        start_idx, end_idx = ranges[t_id]
        t = threading.Thread(
            target=pso_worker,
            args=(
                t_id,
                start_idx,
                end_idx,
                objective,
                positions,
                velocities,
                pbest_positions,
                pbest_values,
                shared_gbest,
                bounds,
                w,
                c1,
                c2,
                barrier_start,
                barrier_end,
                stop_event,
            ),
            daemon=True,  # nie blokuje programu przy nagłym wyjściu
        )
        t.start()
        threads.append(t)

    convergence_window = deque(maxlen=m_no_improve + 1)
    convergence_window.append(float(gbest_value))
    iterations_done = 0

    try:
        for it in range(max_iters):
            # 1. Start iteracji (zwolnienie workerów z bariery)
            barrier_start.wait()

            # 2. Oczekiwanie na koniec obliczeń wątków
            barrier_end.wait()

            # 3. Aktualizacja globalnego GBest – sekwencyjnie na bazie pbestów
            curr_best_idx = np.argmin(pbest_values)
            if pbest_values[curr_best_idx] < shared_gbest[1]:
                shared_gbest[1] = pbest_values[curr_best_idx]
                shared_gbest[0][:] = pbest_positions[curr_best_idx]

            iterations_done = it + 1
            convergence_window.append(float(shared_gbest[1]))

            # Warunki stopu
            if tryb_stopu == "known" and x_opt is not None:
                if stop_znane_optimum(shared_gbest[0], x_opt, eps_opt):
                    break
            elif tryb_stopu == "no_improve":
                if stop_brak_poprawy_deque(convergence_window, eps_no_improve):
                    break

    except threading.BrokenBarrierError:
        # może się zdarzyć przy resetowaniu / kończeniu
        pass
    finally:
        # Sprzątanie
        stop_event.set()
        try:
            barrier_start.reset()
        except:
            pass
        try:
            barrier_end.reset()
        except:
            pass
        for t in threads:
            t.join()

    # Nie zwracamy listy pozycji (pusta lista) – oszczędność RAM
    return shared_gbest[0], shared_gbest[1], iterations_done, {}, []


# ==========================================
# 4. EKSPERYMENTY
# ==========================================

ITERATIONS = 10

def eksperymenty():
    sprawdz_srodowisko()
    funkcje = [
        ("Schwefel", schwefel, (-100.0, 500.0)),
        ("Rosenbrock", rosenbrock, (-10.0, 10.0)),
    ]

    ns = [2]  # możesz zmienić np. na [2, 10, 50] jak w sekwencyjnej

    for nazwa, objective, bounds in funkcje:
        print(f"\n=== Zadanie: {nazwa} (wersja wielowątkowa) ===")

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
                best_x, best_f, iters, metadata, positions = uruchom_pso_threaded(
                    objective=objective,
                    n_dim=n,
                    bounds=bounds,
                    swarm_size=1000,
                    max_iters=10000,
                    tryb_stopu="known",
                    x_opt=x_opt,
                    eps_opt=1e-3,
                    m_no_improve=50,      # nieużywane przy "known", ale zostawiamy
                    eps_no_improve=1e-6,  # nieużywane przy "known"
                    random_state=0,  # możesz dać 0, jeśli chcesz zawsze ten sam bieg
                    n_threads=None,       # None = auto (os.cpu_count)
                )
                end = time.time()

                time_passed = end - start
                iterations.append(iters)
                best_values.append(float(best_f))
                times.append(time_passed)
                times_div_iterations.append(time_passed / iters)

            print(f"[Kryterium 1] iteracje (średnia, odchylenie, min, max): "
                  f"{statistics.mean(iterations)}, {statistics.stdev(iterations)}, "
                  f"{min(iterations)}, {max(iterations)}")
            print(f"[Kryterium 1] najlepsze f(x) (średnia, odchylenie, min, max): "
                  f"{statistics.mean(best_values)}, {statistics.stdev(best_values)}, "
                  f"{min(best_values)}, {max(best_values)}")
            print(f"[Kryterium 1] czas (średnia, odchylenie, min, max): "
                  f"{statistics.mean(times)}, {statistics.stdev(times)}, "
                  f"{min(times)}, {max(times)}")
            print(f"[Kryterium 1] czas / iteracje (średnia, odchylenie, min, max): "
                  f"{statistics.mean(times_div_iterations)}, {statistics.stdev(times_div_iterations)}, "
                  f"{min(times_div_iterations)}, {max(times_div_iterations)}")

            # ---------------------------------------------
            # Kryterium 2 — brak poprawy
            # ---------------------------------------------
            best_values = []
            iterations = []
            times = []
            times_div_iterations = []

            for iteration in range(ITERATIONS):
                # optimum znane nie jest potrzebne dla "no_improve",
                # ale możesz zostawić x_opt jak wyżej, nie przeszkadza.
                start = time.time()
                best_x2, best_f2, iters2, metadata2, positions2 = uruchom_pso_threaded(
                    objective=objective,
                    n_dim=n,
                    bounds=bounds,
                    swarm_size=1000,
                    max_iters=10000,
                    tryb_stopu="no_improve",
                    x_opt=None,
                    eps_opt=1e-3,       # nieużywane przy "no_improve"
                    m_no_improve=50,
                    eps_no_improve=1e-6,
                    random_state=0,
                    n_threads=None,
                )
                end = time.time()

                time_passed = end - start
                iterations.append(iters2)
                best_values.append(float(best_f2))
                times.append(time_passed)
                times_div_iterations.append(time_passed / iters2)

            print(f"[Kryterium 2] iteracje (średnia, odchylenie, min, max): "
                  f"{statistics.mean(iterations)}, {statistics.stdev(iterations)}, "
                  f"{min(iterations)}, {max(iterations)}")
            print(f"[Kryterium 2] najlepsze f(x) (średnia, odchylenie, min, max): "
                  f"{statistics.mean(best_values)}, {statistics.stdev(best_values)}, "
                  f"{min(best_values)}, {max(best_values)}")
            print(f"[Kryterium 2] czas (średnia, odchylenie, min, max): "
                  f"{statistics.mean(times)}, {statistics.stdev(times)}, "
                  f"{min(times)}, {max(times)}")
            print(f"[Kryterium 2] czas / iteracje (średnia, odchylenie, min, max): "
                  f"{statistics.mean(times_div_iterations)}, {statistics.stdev(times_div_iterations)}, "
                  f"{min(times_div_iterations)}, {max(times_div_iterations)}")


if __name__ == "__main__":

    eksperymenty()
