
import numpy as np
import casadi as ca
from solver import Solver, Visualize
def rosenbrock(x, a=1, b=5):
    # Should not be initialized with x = [1,1] -> grad = 0
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2


if __name__ == "__main__":
    Nx = 2
    x_SX = ca.SX.sym('x_SX', Nx)

    obj = ca.Function('obj_func', [x_SX], [rosenbrock(x_SX)])
    solver_setting = {}
    # solver_setting['type'] = ['Newton', 'BFGS_wolfe_condition']
    solver_setting['type'] = ['Newton', 'gradient_descent', 'BFGS_wolfe_condition']
    solver_setting['gradient_descent'] = {}
    solver_setting['gradient_descent']['alpha'] = 0.1 # Oscillate + diverge
    solver_setting['tol_obj_diff'] = 1e-14
    # solver_setting['initial_guess']  = ca.DM([0.6,0.6])
    solver_setting['initial_guess'] = ca.DM([-0.6, 0.6])
    # Solver(obj, solver_setting)
    level = [0.5,1,5, 10,20]
    Visualize(obj, solver_setting, level)

