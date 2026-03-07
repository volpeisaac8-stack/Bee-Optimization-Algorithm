from algorithm import *
from opfunu.cec_based.cec2017 import F12017, F22017, F32017

def cec_objective(x):
    func = F12017(ndim=len(x))
    return [func.evaluate(x)]

best_pos, best_fit, history = adaptive_bee_optimization_live(
    objective_functions=[cec_objective],
    min_bounds=[-100]*10,
    max_bounds=[100]*10,
    num_bees=200,
    max_iterations=500
)