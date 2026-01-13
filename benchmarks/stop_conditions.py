import numpy as np


def stop_znane_optimum(gbest_position, x_opt, eps_opt):

    diff = gbest_position - x_opt
    norm = np.linalg.norm(diff)
    return norm <= eps_opt


def stop_brak_poprawy_deque(history_deque, eps_no_improve):

    if len(history_deque) < history_deque.maxlen:
        return False

    val_old = history_deque[0]
    val_new = history_deque[-1]

    return abs(val_old - val_new) <= eps_no_improve
