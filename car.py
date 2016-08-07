import numpy as np
import utils
import theano as th
import theano.tensor as tt
import theano.tensor.slinalg as ts
from trajectory import Trajectory
import feature

class Car(object):
    def __init__(self, dyn, x0, color='yellow', T=5):
        self.data0 = {'x0': x0}
        self.bounds = [(-1., 1.), (-1., 1.)]
        self.T = T
        self.dyn = dyn
        self.traj = Trajectory(T, dyn)
        self.traj.x0.set_value(x0)
        self.linear = Trajectory(T, dyn)
        self.linear.x0.set_value(x0)
        self.color = color
        self.default_u = np.zeros(self.dyn.nu)
    def reset(self):
        self.traj.x0.set_value(self.data0['x0'])
        self.linear.x0.set_value(self.data0['x0'])
        for t in range(self.T):
            self.traj.u[t].set_value(np.zeros(self.dyn.nu))
            self.linear.u[t].set_value(self.default_u)
    def move(self):
        self.traj.tick()
        self.linear.x0.set_value(self.traj.x0.get_value())
    @property
    def x(self):
        return self.traj.x0.get_value()
    @property
    def u(self):
        return self.traj.u[0].get_value()
    @u.setter
    def u(self, value):
        self.traj.u[0].set_value(value)
    def control(self, steer, gas):
        pass

class UserControlledCar(Car):
    def __init__(self, *args, **vargs):
        Car.__init__(self, *args, **vargs)
        self.bounds = [(-1., 1.), (-1., 1.)]
        self.follow = None
        self.fixed_control = None
        self._fixed_control = None
    def fix_control(self, ctrl):
        self.fixed_control = ctrl
        self._fixed_control = ctrl
    def control(self, steer, gas):
        if self.fixed_control is not None:
            self.u = self.fixed_control[0]
            print self.fixed_control[0]
            if len(self.fixed_control)>1:
                self.fixed_control = self.fixed_control[1:]
        elif self.follow is None:
            self.u = [steer, gas]
        else:
            u = self.follow.u[0].get_value()
            if u[1]>=1.:
                u[1] = 1.
            if u[1]<=-1.:
                u[1] = -1.
            self.u = u
    def reset(self):
        Car.reset(self)
        self.fixed_control = self._fixed_control

class SimpleOptimizerCar(Car):
    def __init__(self, *args, **vargs):
        Car.__init__(self, *args, **vargs)
        self.bounds = [(-1., 1.), (-1., 1.)]
        self.cache = []
        self.index = 0
        self.sync = lambda cache: None
    def reset(self):
        Car.reset(self)
        self.index = 0
    @property
    def reward(self):
        return self._reward
    @reward.setter
    def reward(self, reward):
        self._reward = reward+100.*feature.bounded_control(self.bounds)
        self.optimizer = None
    def control(self, steer, gas):
        print len(self.cache)
        if self.index<len(self.cache):
            self.u = self.cache[self.index]
        else:
            if self.optimizer is None:
                r = self.traj.reward(self.reward)
                self.optimizer = utils.Maximizer(r, self.traj.u)
            self.optimizer.maximize()
            self.cache.append(self.u)
            self.sync(self.cache)
        self.index += 1

class NestedOptimizerCar(Car):
    def __init__(self, *args, **vargs):
        Car.__init__(self, *args, **vargs)
        self.bounds = [(-3., 3.), (-2., 2.)]
    @property
    def human(self):
        return self._human
    @human.setter
    def human(self, value):
        self._human = value
        self.traj_h = Trajectory(self.T, self.human.dyn)
    def move(self):
        Car.move(self)
        self.traj_h.tick()
    @property
    def rewards(self):
        return self._rewards
    @rewards.setter
    def rewards(self, vals):
        self._rewards = vals
        self.optimizer = None
    def control(self, steer, gas):
        if self.optimizer is None:
            reward_h, reward_r = self.rewards
            reward_h = self.traj_h.reward(reward_h)
            reward_r = self.traj.reward(reward_r)
            self.optimizer = utils.NestedMaximizer(reward_h, self.traj_h.u, reward_r, self.traj.u)
        self.traj_h.x0.set_value(self.human.x)
        self.optimizer.maximize(bounds = self.bounds)
