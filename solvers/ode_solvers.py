from tqdm import tqdm

class ODESolve:
    def __init__(self, solve_method = "euler"):

        methods = {
            "euler": self.euler_solver,
            "rk4": self.rk4_solver
        }

        if solve_method not in methods:
            raise ValueError(f"Method {solve_method} not supported. Choose from {list(methods.keys())}")
        
        self.solver = methods[solve_method]

    def __call__(self, *args, **kwargs):
        return self.solver(*args, **kwargs)

    @staticmethod
    def euler_solver(model, x, embeddings, num_steps, mode='forward', init_time=0):
        """
        Euler Solver for velocity models.
        
        Args:
            model: A callable (nn.Module) that computes dx/dt = f(x, t).
            x: Initial state tensor.
            num_steps: Number of integration steps.
            mode: 'forward' (0 -> 1) or 'backward' (1 -> 0).
        """
        dt = (1-init_time) / num_steps
        
        # Adjust direction for backward mode
        if mode == 'backward':
            t = 1.0
            dt = -dt
        else:
            t = init_time

        for _ in tqdm(range(num_steps), desc=f"Euler {mode}"):
            # x_{n+1} = x_n + dt * f(t_n, x_n)
            grad = model.velocity(x, t, embeddings)
            x = x + dt * grad
            
            # Update time
            t += dt
            
        return x
    
    @staticmethod
    def rk4_solver(model, x, embeddings, num_steps, mode='forward', init_time=0):
        """
        4th order Runge-Kutta (RK4) Solver for velocity models.
        
        Args:
            model: A callable (nn.Module) that computes dx/dt = f(t, x).
            x: Initial state tensor.
            num_steps: Number of integration steps.
            mode: 'forward' (0 -> 1) or 'backward' (1 -> 0).
        """
        dt = 1.0 / num_steps
        
        # If backward, we reverse the time direction and the sign of dt
        if mode == 'backward':
            t = 1.0
            dt = -dt
        else:
            t = init_time

        for _ in tqdm(range(num_steps), desc=f"RK4 {mode}"):
            # k1 = f(t, x)
            k1 = model.velocity(x, t, embeddings)
            
            # k2 = f(t + dt/2, x + dt/2 * k1)
            k2 = model.velocity(x + (dt/2) * k1, t + dt/2, embeddings)
            
            # k3 = f(t + dt/2, x + dt/2 * k2)
            k3 = model.velocity(x + (dt/2) * k2, t + dt/2, embeddings)
            
            # k4 = f(t + dt, x + dt * k3)
            k4 = model.velocity(x + dt * k3, t + dt, embeddings)
            
            # Update state: x = x + dt/6 * (k1 + 2k2 + 2k3 + k4)
            x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            # Update time
            t += dt
            
        return x


