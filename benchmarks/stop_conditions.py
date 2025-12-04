import numpy as np


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
