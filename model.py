import numpy as np


class DiscreteModel:

    def __init__(self, items, probs):
        if len(items) != len(probs):
            raise ValueError("length of items and probability must match")

        self.prob_dict = dict()
        for item, prob in zip(items, probs):
            self.prob_dict[item] = prob

    def update(self, data, hypothesis, likelihood):
        self.prob_dict[hypothesis] *= likelihood(data, hypothesis)

    def normalize(self):
        total = 0.0
        for hypothesis in self.prob_dict:
            total += self.prob_dict[hypothesis]
        for hypothesis in self.prob_dict:
            self.prob_dict[hypothesis] /= total


class GridInfo:

    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps


class Grid4DModel(DiscreteModel):

    # domain: array of GridInfo()
    def __init__(self, domain):
        if len(domain) != 4:
            raise ValueError("This is 4D Model")

        tuples = []
        for a in np.linspace(domain[0].start, domain[0].end, num=domain[0].steps):
            for b in np.linspace(domain[1].start, domain[1].end, num=domain[1].steps):
                for c in np.linspace(domain[2].start, domain[2].end, num=domain[2].steps):
                    for d in np.linspace(domain[3].start, domain[3].end, num=domain[3].steps):
                        tuples.append((a, b, c, d))
        probs = [1 for i in range(len(tuples))]

        DiscreteModel.__init__(self, tuples, probs)

    def marginal_distribution(self, dim):
        if dim > 3:
            raise ValueError("This is 4D Model")
        if dim < 0:
            raise ValueError("No Minus")
        marginal = {}
        for tuple_grid in self.prob_dict.keys():
            if tuple_grid[dim] in marginal:
                marginal[tuple_grid[dim]] += self.prob_dict[tuple_grid]
            else:
                marginal[tuple_grid[dim]] = self.prob_dict[tuple_grid]
        marginal_tuple = [(k, v) for k, v in marginal.items()]
        marginal_tuple.sort(key=lambda x: x[0])
        marginal_np = np.zeros(shape=(len(marginal_tuple), 2))
        for i in range(len(marginal_tuple)):
            marginal_np[i, :] = marginal_tuple[i]
        return marginal_np
