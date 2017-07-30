import numpy as np


class OdeSimulator:

    def __init__(self, odesolver, stdev=1.0):
        self.odesolver = odesolver
        self.stdev = stdev
        self.odesolver.solve()

    def simulate(self, t):
        return self.odesolver.value(t) + np.random.normal(scale=self.stdev)
