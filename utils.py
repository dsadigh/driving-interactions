import theano as th
import theano.tensor as tt
import theano.tensor.slinalg as ts
import scipy.optimize
import numpy as np
import time

def extract(var):
    return th.function([], var, mode=th.compile.Mode(linker='py'))()

def shape(var):
    return extract(var.shape)

def vector(n):
    return th.shared(np.zeros(n))

def matrix(n, m):
    return tt.shared(np.zeros((n, m)))

def grad(f, x, constants=[]):
    ret = th.gradient.grad(f, x, consider_constant=constants, disconnected_inputs='warn')
    if isinstance(ret, list):
        ret = tt.concatenate(ret)
    return ret

def jacobian(f, x, constants=[]):
    sz = shape(f)
    return tt.stacklists([grad(f[i], x) for i in range(sz)])
    ret = th.gradient.jacobian(f, x, consider_constant=constants)
    if isinstance(ret, list):
        ret = tt.concatenate(ret, axis=1)
    return ret

def hessian(f, x, constants=[]):
    return jacobian(grad(f, x, constants=constants), x, constants=constants)

class NestedMaximizer(object):
    def __init__(self, f1, vs1, f2, vs2):
        self.f1 = f1
        self.f2 = f2
        self.vs1 = vs1
        self.vs2 = vs2
        self.sz1 = [shape(v)[0] for v in self.vs1]
        self.sz2 = [shape(v)[0] for v in self.vs2]
        for i in range(1, len(self.sz1)):
            self.sz1[i] += self.sz1[i-1]
        self.sz1 = [(0 if i==0 else self.sz1[i-1], self.sz1[i]) for i in range(len(self.sz1))]
        for i in range(1, len(self.sz2)):
            self.sz2[i] += self.sz2[i-1]
        self.sz2 = [(0 if i==0 else self.sz2[i-1], self.sz2[i]) for i in range(len(self.sz2))]
        self.df1 = grad(self.f1, vs1)
        self.new_vs1 = [tt.vector() for v in self.vs1]
        self.func1 = th.function(self.new_vs1, [-self.f1, -self.df1], givens=zip(self.vs1, self.new_vs1))
        def f1_and_df1(x0):
            return self.func1(*[x0[a:b] for a, b in self.sz1])
        self.f1_and_df1 = f1_and_df1
        J = jacobian(grad(f1, vs2), vs1)
        H = hessian(f1, vs1)
        g = grad(f2, vs1)
        self.df2 = -tt.dot(J, ts.solve(H, g))+grad(f2, vs2)
        self.func2 = th.function([], [-self.f2, -self.df2])
        def f2_and_df2(x0):
            for v, (a, b) in zip(self.vs2, self.sz2):
                v.set_value(x0[a:b])
            self.maximize1()
            return self.func2()
        self.f2_and_df2 = f2_and_df2
    def maximize1(self):
        x0 = np.hstack([v.get_value() for v in self.vs1])
        opt = scipy.optimize.fmin_l_bfgs_b(self.f1_and_df1, x0=x0)[0]
        for v, (a, b) in zip(self.vs1, self.sz1):
            v.set_value(opt[a:b])
    def maximize(self, bounds={}):
        t0 = time.time()
        if not isinstance(bounds, dict):
            bounds = {v: bounds for v in self.vs2}
        B = []
        for v, (a, b) in zip(self.vs2, self.sz2):
            if v in bounds:
                B += bounds[v]
            else:
                B += [(None, None)]*(b-a)
        x0 = np.hstack([v.get_value() for v in self.vs2])
        def f(x0):
            #if time.time()-t0>60:
             #   raise Exception('Too long')
            return self.f2_and_df2(x0)
        opt = scipy.optimize.fmin_l_bfgs_b(f, x0=x0, bounds=B)
        diag = opt[2]['task']
        opt = opt[0]
        for v, (a, b) in zip(self.vs2, self.sz2):
            v.set_value(opt[a:b])
        self.maximize1()

class Maximizer(object):
    def __init__(self, f, vs, g={}, pre=None, gen=None, method='bfgs', eps=1, iters=100000, debug=False, inf_ignore=np.inf):
        self.inf_ignore = inf_ignore
        self.debug = debug
        self.iters = iters
        self.eps = eps
        self.method = method
        def one_gen():
            yield
        self.gen = gen
        if self.gen is None:
            self.gen = one_gen
        self.pre = pre
        self.f = f
        self.vs = vs
        self.sz = [shape(v)[0] for v in self.vs]
        for i in range(1,len(self.sz)):
            self.sz[i] += self.sz[i-1]
        self.sz = [(0 if i==0 else self.sz[i-1], self.sz[i]) for i in range(len(self.sz))]
        if isinstance(g, dict):
            self.df = tt.concatenate([g[v] if v in g else grad(f, v) for v in self.vs])
        else:
            self.df = g
        self.new_vs = [tt.vector() for v in self.vs]
        self.func = th.function(self.new_vs, [-self.f, -self.df], givens=zip(self.vs, self.new_vs))
        def f_and_df(x0):
            if self.debug:
                print x0
            s = None
            N = 0
            for _ in self.gen():
                if self.pre:
                    for v, (a, b) in zip(self.vs, self.sz):
                        v.set_value(x0[a:b])
                    self.pre()
                res = self.func(*[x0[a:b] for a, b in self.sz])
                if np.isnan(res[0]).any() or np.isnan(res[1]).any() or (np.abs(res[0])>self.inf_ignore).any() or (np.abs(res[1])>self.inf_ignore).any():
                    continue
                if s is None:
                    s = res
                    N = 1
                else:
                    s[0] += res[0]
                    s[1] += res[1]
                    N += 1
            s[0]/=N
            s[1]/=N
            return s
        self.f_and_df = f_and_df
    def argmax(self, vals={}, bounds={}):
        if not isinstance(bounds, dict):
            bounds = {v: bounds for v in self.vs}
        B = []
        for v, (a, b) in zip(self.vs, self.sz):
            if v in bounds:
                B += bounds[v]
            else:
                B += [(None, None)]*(b-a)
        x0 = np.hstack([np.asarray(vals[v]) if v in vals else v.get_value() for v in self.vs])
        if self.method=='bfgs':
            opt = scipy.optimize.fmin_l_bfgs_b(self.f_and_df, x0=x0, bounds=B)[0]
        elif self.method=='gd':
            opt = x0
            for _ in range(self.iters):
                opt -= self.f_and_df(opt)[1]*self.eps
        else:
            opt = scipy.optimize.minimize(self.f_and_df, x0=x0, method=self.method, jac=True).x
        return {v: opt[a:b] for v, (a, b) in zip(self.vs, self.sz)}
    def maximize(self, *args, **vargs):
        result = self.argmax(*args, **vargs)
        for v, res in result.iteritems():
            v.set_value(res)

if __name__ == '__main__':
    x = vector(1)
    y = vector(1)
    f = -(x[0]-y[0])**2
    def gen():
        for i in range(10):
            y.set_value([i])
            yield
    y.set_value([10.])
    optimizer = Maximizer(f, [x], gen=gen, method='CG')
    optimizer.maximize()
    print x.get_value()
    quit()
    x1 = vector(2)
    x2 = vector(1)
    f1 = -((x1[0]-x2[0]-1)**2+(x1[1]-x2[0])**2)-100.*tt.exp(40.*(x1[0]-4))
    f2 = -((x1[0]-2.)**2+(x1[1]-4.)**2)-(x2[0]-6.)**2
    optimizer = NestedMaximizer(f1, [x1], f2, [x2])
    optimizer.maximize(bounds=[(0., 10.)])
    print x2.get_value()
    print x1.get_value()
