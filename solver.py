import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import inspect
from matplotlib import cm


class Model:
    def __init__(self, obj_fn):
        self.obj_fn = obj_fn
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
    def __init__(self, obj_fn, solver_setting):
        super().__init__(obj_fn)

        self.solver_setting = solver_setting

        if 'initial_guess' not in solver_setting:
            self.solver_setting['initial_guess'] = ca.DM.rand(2,1) - 0.5
        if 'max_iter' not in solver_setting:
            self.solver_setting['max_iter'] = 300
        if 'tol_obj_diff' not in solver_setting:
            self.solver_setting['tol_obj_diff'] = 1e-8

        self.x_dict = {}
        self.obj_dict = {}
        self.iter_dict = {}

        for type in solver_setting['type']:
            if type == 'Newton':
                if 'Newton' not in solver_setting:
                    self.solver_setting['Newton'] = None
                # else:
                    # TODO: parameter setting
                self.Newton()

            elif type == 'gradient_descent':
                if 'gradient_descent' not in solver_setting:
                    self.solver_setting['gradient_descent'] = None
                    self.alpha = 0.1
                else:
                    self.alpha = solver_setting['gradient_descent']['alpha']
                self.gradient_descent()


            elif type == 'BFGS_wolfe_condition':
                #TODO: This should be modified to be compatible with other conditions, e.g. Wolfe and Goldstein condition
                if 'BFGS_wolfe_condition' not in solver_setting:
                    self.solver_setting['BFGS_wolfe_condition'] = None
                    self.c1 = 0.4
                    self.c2 = 0.9 #  For a Newton or quasi-Newton method, choose c2 = 0.9
                else:
                    self.c1 = solver_setting['BFGS_wolfe_condition']['c1']
                    self.c2 = solver_setting['BFGS_wolfe_condition']['c2']
                self.BFGS_wolfe_condition()

            elif type == 'Fletcher_Reeves':
                if 'Fletcher_Reeves' not in solver_setting:
                    self.solver_setting['Fletcher_Reeves'] = None
                    self.alpha = 0.1
                    self.c1 = 0.4
                    self.c2 = 0.45 #  For FLETCHER–REEVES method, choose c2 < 0.5
                else:
                    self.alpha = solver_setting['Fletcher_Reeves']['alpha']
                    self.c1 = solver_setting['Fletcher_Reeves']['c1']
                    self.c2 = solver_setting['Fletcher_Reeves']['c2']
                self.Fletcher_Reeves()


            else:
                print('solver type error')


    def gradient_descent(self):
        alpha = self.alpha
        obj_list = []
        x_list = []

        obj_fn = self.obj_fn
        grad_func = self.grad_func
        # hessian_func = self.hessian_func

        initial_guess = self.solver_setting['initial_guess']
        max_iter = self.solver_setting['max_iter']
        tol_obj_diff = self.solver_setting['tol_obj_diff']
        error = 1e5

        xi = initial_guess
        iter_counter = 0

        obj_list += obj_fn(xi).full().flatten().tolist()
        x_list += xi.full().flatten().tolist()
        for i in range(max_iter):
            if error >= tol_obj_diff:
                iter_counter += 1
                grad_i = grad_func(xi)
                dx = - alpha * grad_i / (ca.norm_2(grad_i))
                xi_prev = xi
                xi = xi + dx
                error = ca.norm_2(xi - xi_prev)
                obj_list += obj_fn(xi).full().flatten().tolist()
                x_list += xi.full().flatten().tolist()
            else:
                break
        this_function_name = inspect.currentframe().f_code.co_name

        self.x_dict[this_function_name] = x_list
        self.obj_dict[this_function_name] = obj_list
        self.iter_dict[this_function_name] = iter_counter


    def Newton(self):
        obj_list = []
        x_list = []

        obj_fn = self.obj_fn
        grad_func = self.grad_func
        hessian_func = self.hessian_func

        initial_guess = self.solver_setting['initial_guess']
        max_iter = self.solver_setting['max_iter']
        tol_obj_diff = self.solver_setting['tol_obj_diff']
        error = 1e5

        xi = initial_guess
        iter_counter = 0

        obj_list += obj_fn(xi).full().flatten().tolist()
        x_list += xi.full().flatten().tolist()
        for i in range(max_iter):
            if error >= tol_obj_diff:
                iter_counter += 1
                hessian_i = hessian_func(xi)
                grad_i = grad_func(xi)
                hessian_i_inv = ca.solve(hessian_i, ca.DM.eye(hessian_i.size1()))
                # print(grad_i,hessian_i)
                dx = - hessian_i_inv @ grad_i
                xi_prev = xi
                xi = xi + dx
                error = ca.norm_2(xi - xi_prev)
                obj_list += obj_fn(xi).full().flatten().tolist()
                x_list += xi.full().flatten().tolist()
            else:
                break
        this_function_name = inspect.currentframe().f_code.co_name

        self.x_dict[this_function_name] = x_list
        self.obj_dict[this_function_name] = obj_list
        self.iter_dict[this_function_name] = iter_counter

    def strong_Wolfe_condition(self, xk, pk):
        alpha = 1
        c1 = self.c1
        c2 = self.c2
        obj_fn = self.obj_fn
        grad_func = self.grad_func
        while (True):
            if alpha <= 1e-9:
                return alpha
            cond_armijo = (obj_fn(xk + alpha * pk) <= obj_fn(xk) + c1 * alpha * grad_func(xk).T @ pk)
            cond_curvature = ca.fabs((grad_func(xk + alpha * pk).T @ pk) <= c2 * ca.fabs(grad_func(xk).T @ pk))
            if cond_armijo == True and cond_curvature == True:
                return alpha
            else:
                alpha = alpha * 0.95

    def Wolfe_condition(self, xk, pk):
        alpha = 1
        c1 = self.c1
        c2 = self.c2
        obj_fn = self.obj_fn
        grad_func = self.grad_func

        while (True):
            if alpha <= 1e-9:
                return alpha
            cond_armijo = (obj_fn(xk + alpha * pk) <= obj_fn(xk) + c1 * alpha * grad_func(xk).T @ pk)
            cond_curvature = (grad_func(xk + alpha * pk).T @ pk >= c2 * grad_func(xk).T @ pk)
            if cond_armijo == True and cond_curvature == True:
                return alpha
            else:
                alpha = alpha * 0.95

    def plot_phi(self, xk, pk, alpha_sel):
        obj_fn = self.obj_fn
        alpha = np.arange(0, 1, 0.01)
        N_alpha = np.shape(alpha)[0]

        phi_list = []

        for i in range(N_alpha):
            alpha_i = alpha[i]
            phi_list += [obj_fn(xk + alpha_i * pk).full().flatten().tolist()]

        fig, ax = plt.subplots()
        ax.plot(alpha, phi_list)
        ax.scatter(alpha_sel, obj_fn(xk + alpha_sel * pk).full().flatten().tolist(), s=50, color='r', zorder=10,
                   label='alpha =' + str(alpha_sel))
        ax.legend()
        fig.show()

    def BFGS_wolfe_condition(self):
        # c1 = self.c1
        # c2 = self.c2

        obj_list = []
        x_list = []

        obj_fn = self.obj_fn
        grad_func = self.grad_func
        hessian_func = self.hessian_func

        initial_guess = self.solver_setting['initial_guess']
        max_iter = self.solver_setting['max_iter']
        tol_obj_diff = self.solver_setting['tol_obj_diff']
        error = 1e5
        I = ca.diag([1] * self.Nx)
        xi = initial_guess
        hessian_i = hessian_func(xi)
        H0 = I
        Hi = H0

        iter_counter = 0
        obj_list += obj_fn(xi).full().flatten().tolist()
        x_list += xi.full().flatten().tolist()
        # max_iter = 20
        for i in range(max_iter):
            if error >= tol_obj_diff:
                iter_counter += 1
                grad_i = grad_func(xi)
                pi = - Hi @ grad_i
                alpha = self.Wolfe_condition(xi, pi)
                xi_prev = xi
                # print(alpha, pi)
                # self.plot_phi(xi, pi, alpha) # This line plot value function according to different alpha
                xi = xi + alpha * pi
                error = ca.norm_2(xi - xi_prev)
                obj_list += obj_fn(xi).full().flatten().tolist()
                x_list += xi.full().flatten().tolist()
                grad_plus = grad_func(xi)
                si = xi - xi_prev
                yi = grad_plus - grad_i
                rho_i = 1 / (yi.T @ si)
                Hi = (I - rho_i * si @ yi.T) @ Hi @ (I - rho_i * si @ yi.T) + rho_i @ si @ si.T
            else:
                break
        this_function_name = inspect.currentframe().f_code.co_name

        self.x_dict[this_function_name] = x_list
        self.obj_dict[this_function_name] = obj_list
        self.iter_dict[this_function_name] = iter_counter

        # print(x_list)

    def Fletcher_Reeves(self):
        alpha = self.alpha
        # print(alpha)
        obj_list = []
        x_list = []

        obj_fn = self.obj_fn
        grad_func = self.grad_func
        # hessian_func = self.hessian_func

        initial_guess = self.solver_setting['initial_guess']
        max_iter = self.solver_setting['max_iter']
        tol_obj_diff = self.solver_setting['tol_obj_diff']
        error = 1e5
        xi = initial_guess
        iter_counter = 0
        obj_list += obj_fn(xi).full().flatten().tolist()
        x_list += xi.full().flatten().tolist()
        # max_iter = 20
        obj_i = obj_fn(xi)
        grad_i = grad_func(xi)
        pi = - grad_i
        for i in range(max_iter):
            if error >= tol_obj_diff:
                iter_counter += 1
                xi_prev = xi
                alpha = self.Wolfe_condition(xi, pi)
                x_next = xi + alpha * pi
                grad_plus = grad_func(x_next)
                beta_i_next = (grad_plus.T @ grad_plus ) / (grad_i.T @ grad_i)
                pk_next = - grad_plus + beta_i_next * pi

                grad_i = grad_plus
                pi = pk_next
                xi = x_next

                error = ca.norm_2(xi - xi_prev)
                obj_list += obj_fn(xi).full().flatten().tolist()
                x_list += xi.full().flatten().tolist()
            else:
                break
        this_function_name = inspect.currentframe().f_code.co_name

        self.x_dict[this_function_name] = x_list
        self.obj_dict[this_function_name] = obj_list
        self.iter_dict[this_function_name] = iter_counter
        # print(x_list)


class Visualize(Solver):
    def __init__(self, obj_fn, solver_setting, level = 10):
        super().__init__(obj_fn, solver_setting)

        self.plot_init() # TODO: This step should might be a bit slow.
        # 3D visualization
        fig, ax = self.plot3D_init()
        self.fig = fig
        self.ax = ax

        N_solver = 0
        for type in self.solver_setting['type']:
            # obj_list = exec(" self." + type + "_obj_list")
            # x_list =  exec("self." + type + "_x_list")
            # N_iter = exec(" self." + type + "_iter_counter")
            obj_list = self.obj_dict[type]
            x_list = self.x_dict[type]
            N_iter = self.iter_dict[type]
            self.update_plot3D(N_iter, x_list, obj_list, N_solver, type)
            N_solver += 1
        fig.show()

        # 2D visualization
        fig, ax = self.plot2D_init(level = level)  #
        self.fig = fig
        self.ax = ax

        N_solver = 0
        for type in self.solver_setting['type']:
            # obj_list = self.obj_dict[type]
            x_list = self.x_dict[type]
            N_iter = self.iter_dict[type]
            self.update_plot2D(N_iter, x_list, N_solver, type)
            N_solver += 1
        fig.show()

        # Convergence rate
        fig, ax = self.plot_convergence_init()
        self.fig = fig
        self.ax = ax

        N_solver = 0
        for type in self.solver_setting['type']:
            obj_list = self.obj_dict[type]
            # x_list = self.x_dict[type]
            N_iter = self.iter_dict[type]
            self.update_plot_convergence(N_iter, obj_list, N_solver, type)
            N_solver += 1
        fig.show()



    def plot_init(self, x1_range=None, x2_range=None, z_range=None):
        if x1_range is None:
            x1_val = np.arange(-2, 2, 0.1, dtype=np.float32)
        else:
            x1_val = x1_range
        if x2_range is None:
            x2_val = np.arange(-2, 2, 0.1, dtype=np.float32)
        else:
            x2_val = x2_range
        x1_val_mesh, x2_val_mesh = np.meshgrid(x1_val, x2_val)
        x1_val_mesh_flat = x1_val_mesh.reshape([1, -1])
        x2_val_mesh_flat = x2_val_mesh.reshape([1, -1])

        z_val_mesh_flat = np.zeros_like(x1_val_mesh_flat)
        self.cost_for(x1_val_mesh_flat, x2_val_mesh_flat, z_val_mesh_flat)
        z_val_mesh = np.reshape(z_val_mesh_flat, (x1_val_mesh.shape))

        self.x1_val_mesh = x1_val_mesh
        self.x2_val_mesh = x2_val_mesh
        # self.x1_val_mesh_flat = x1_val_mesh_flat
        # self.x2_val_mesh_flat = x2_val_mesh_flat
        # self.z_val_mesh_flat = z_val_mesh_flat
        self.z_val_mesh = z_val_mesh

    def cost_for(self, x1, x2, z):
        Nx_p = np.shape(x1)[1]
        x_plot = np.vstack((x1, x2))
        for i in range(Nx_p):
            # print(self.obj_fn(x_plot[:, i]).full().item())
            z[:, i] = self.obj_fn(x_plot[:, i]).full().item()

    def plot_convergence_init(self):
        fig, ax = plt.subplots()
        return fig, ax

    def update_plot_convergence(self, N_iter, obj_list, N_solver, type):
        ax = self.ax
        obj_last = obj_list[-1]
        iter_list = [i for i in range(N_iter)]
        obj_array = np.abs(np.array(obj_list[:-1]) - obj_last)
        ax.set_yscale('log')
        ax.plot(iter_list, obj_array, label=type, color='C' + str(N_solver))
        ax.legend()

    def plot2D_init(self, level = 10):
        fig, ax = plt.subplots()
        x1_val_mesh = self.x1_val_mesh
        x2_val_mesh = self.x2_val_mesh
        z_val_mesh = self.z_val_mesh
        CS = ax.contour(x1_val_mesh, x2_val_mesh, z_val_mesh, level)
        ax.clabel(CS, inline=1, fontsize=10)
        return fig, ax

    def update_plot2D(self, N_iter, x_list, N_solver, type):
        ax = self.ax
        # We assume here Nx = 2
        # Nx = self.Nx
        # N_plot = N_iter + 1
        x1_list = x_list[::2]
        x2_list = x_list[1::2]
        # print(x1_list, x2_list)
        # print(x1_list[0], x2_list[0])
        # ax.scatter(x1_list[0], x2_list[1], s=20, color='C' + str(N_solver))
        ax.plot(x1_list, x2_list, label=type, color='C' + str(N_solver))
        ax.scatter(x1_list[0], x2_list[0], s=20, color='C' + str(N_solver))
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.legend()

    def plot3D_init(self):
        x1_val_mesh = self.x1_val_mesh
        x2_val_mesh = self.x2_val_mesh
        z_val_mesh = self.z_val_mesh

        fig = plt.figure(figsize=(6, 4))
        spec = fig.add_gridspec(nrows=1, ncols=1)
        ax = fig.add_subplot(spec[0, 0], projection='3d')
        ax.plot_surface(x1_val_mesh, x2_val_mesh, z_val_mesh, alpha=0.5, cmap=cm.coolwarm)

        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Z')
        return fig, ax

    def update_plot3D(self, N_iter, x_list, obj_list, N_solver, type):
        ax = self.ax
        Nx = self.Nx
        N_plot = N_iter + 1
        f_value = obj_list[0]
        plot_cache = None
        # print(N_iter, x_list, obj_list)
        for n_plot in range(N_plot):
            xi = x_list[Nx * n_plot:Nx * (n_plot + 1)]
            # print(xi)
            if n_plot == 0:
                # plot_cache = ax.scatter(xi[0], xi[1], f_value, label=type, s=20, depthshade=True, color='C' + str(N_solver))
                ax.scatter(xi[0], xi[1], f_value, label=type, s=20, depthshade=True,
                                        color='C' + str(N_solver))
                x_past = xi
                f_past = f_value
            else:
                f_value = obj_list[n_plot]
                # plot_cache.remove()
                # print(xi)
                # plot_cache = ax.scatter(xi[0], xi[1], f_value, s=1, depthshade=True, color='r')
                ax.plot([x_past[0], xi[0]], [x_past[1], xi[1]], [f_past, f_value], linewidth=2, color='C' + str(N_solver))
                x_past = xi
                f_past = f_value
        ax.legend()