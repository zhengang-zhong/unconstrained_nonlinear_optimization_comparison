
import numpy as np
import casadi as ca
from solver import Solver
def rosenbrock(x, a=1, b=5):
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2


if __name__ == "__main__":
    Nx = 2
    x_SX = ca.SX.sym('x_SX', Nx)

    obj = ca.Function('obj_func', [x_SX], [rosenbrock(x_SX)])
    solver_setting = {}
    solver_setting['type'] = ['Newton', 'Steepest_descent']
    solver_setting['tol_obj_diff'] = 1e-8
    solver_setting['initial_guess']  = ca.DM([1,1])
    # for i in range(10):
        # print(ca.DM.rand(2, 1) - 0.5)
    # Solver(ode, solver_setting)
    H, g = ca.hessian(4 * x_SX[0]**2 + 0.03* x_SX[1]**2 + 2* x_SX[0] * x_SX[1], x_SX)

# def concat(*args):
#     print(len(args))
#     # return print(args)
#
#     solver_opt = {}
#     solver_opt['print_time'] = False
#     solver_opt['ipopt'] = {
#         'max_iter': 500,
#         'print_level': 3,
#         'acceptable_tol': 1e-6,
#         'acceptable_obj_change_tol': 1e-6
#     }
#
# concat("1","2","3")

