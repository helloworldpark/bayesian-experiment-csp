import numpy as np
import matplotlib.pyplot as plot
from model import *
from diffsolver import *


class PredatorPrey(OdeSolver):
    def __init__(self, time_info, initial_values, a, b, c, d):
        OdeSolver.__init__(self, time_info, initial_values)
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def equation(self, t, x):
        x1 = self.a * x[0] - self.b * x[0] * x[1]
        x2 = -self.c * x[1] + self.d * x[0] * x[1]
        return np.array([x1, x2])


class Estimator(Grid4DModel):

    # abcd: tuple of ranges of parameters
    # steps: steps
    # initial_value: tuple of initial value
    def __init__(self, abcd, steps, initial_value, known_error):
        grid_a = GridInfo(start=0.0, end=abcd[0], steps=steps)
        grid_b = GridInfo(start=0.0, end=abcd[1], steps=steps)
        grid_c = GridInfo(start=0.0, end=abcd[2], steps=steps)
        grid_d = GridInfo(start=0.0, end=abcd[3], steps=steps)

        Grid4DModel.__init__(self, [grid_a, grid_b, grid_c, grid_d])

        self.abcd = abcd
        self.initial_value = np.array([initial_value[0], initial_value[1]])
        self.known_error = known_error

    # data: [0] = time, [1,2] = observed value
    def likelihood(self, data, hypothesis):
        time_info = OdeTime(0.0, data[0], 1000)
        solver = PredatorPrey(time_info, self.initial_value, hypothesis[0], hypothesis[1], hypothesis[2], hypothesis[3])
        solver.solve()
        y = solver.ode_output.results[-1]
        diff = y - data[1:2]
        diff = -(diff**2)
        diff = diff.sum() / (2.0 * (self.known_error ** 2))
        return np.exp(-diff)
