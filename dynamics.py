import theano as th
import theano.tensor as tt

class Dynamics(object):
    def __init__(self, nx, nu, f, dt=None):
        self.nx = nx
        self.nu = nu
        self.dt = dt
        if dt is None:
            self.f = f
        else:
            self.f = lambda x, u: x+dt*f(x, u)
    def __call__(self, x, u):
        return self.f(x, u)

class CarDynamics(Dynamics):
    def __init__(self, dt=0.1, ub=[(-3., 3.), (-1., 1.)], friction=1.):
        def f(x, u):
            return tt.stacklists([
                x[3]*tt.cos(x[2]),
                x[3]*tt.sin(x[2]),
                x[3]*u[0],
                u[1]-x[3]*friction
            ])
        Dynamics.__init__(self, 4, 2, f, dt)

if __name__ == '__main__':
    dyn = CarDynamics(0.1)
    x = tt.vector()
    u = tt.vector()
    dyn(x, u)
