import numpy as np


class Simulatable:

    def simulate(self, t):
        raise NotImplementedError("Method simulate is not implemented")


class Simulator:

    def __init__(self, simulatable, stdev=1.0):
        self.simulatable = simulatable
        self.stdev = stdev

    def simulate(self, t):
        return self.simulatable.simulate(t) + np.random.normal(scale=self.stdev)
