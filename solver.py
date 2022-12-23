import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

class Model:
    def __init__(self, obj_fn):
        self.model_fn = obj_fn
        self.Nx = obj_fn.sx_in()[0].shape[0]
        x_SX = ca.SX.sym("x_SX", self.Nx)
        self.x_SX = x_SX

        obj = obj_fn(x_SX)
        H, g = ca.hessian(obj, x_SX)
        grad_func = ca.Function("grad_func", [x_SX], [g])
        hessian_func = ca.Function("hessian_func", [x_SX], [H])

        self.grad_func = grad_func
        self.hessian_func = hessian_func

class Solver(Model):
    def __init__(self, model_fn, solver_setting):
        super().__init__(model_fn)

        if 'initial_guess' not in solver_setting:
            self.solver_setting['initial_guess'] = ca.DM.rand(2,1) - 0.5
        if 'max_iter' not in solver_setting:
            self.solver_setting['max_iter'] = 300

        for type in solver_setting['type']:
            if type == 'Newton':
                if 'Newton' not in solver_setting:
                    self.solver_setting['Newton'] = None
                self.Newton()

            elif type == 'Steepest_descent':
                if 'Steepest_descent' not in solver_setting:
                    self.solver_setting['Steepest_descent'] = None
                self.Steepest_descent()

            else:
                print('solver type error')


    def Steepest_descent(self):
        model_fn = self.model_fn

    def Newton(self):
        model_fn = self.model_fn
        grad_func = self.grad_func
        hessian_func = self.hessian_func

        initial_guess = self.solver_setting['initial_guess']
        max_iter = self.solver_setting['max_iter']


        grad_func()






# class Visualize(Solver):