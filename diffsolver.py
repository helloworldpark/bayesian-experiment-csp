import numpy as np


class OdeTime:
    
    def __init__(self, t_from, t_to, t_intervals):
        self.t_from = t_from
        self.t_to = t_to
        self.t_intervals = t_intervals
        self.dt = (t_to - t_from) / float(t_intervals)

    def time(self, i):
        return self.t_from + self.dt * float(i)


class OdeInput:
    
    #input
    #    time: odeTime
    #    initialValues: numpy matrix of dimension 1*n
    
    def __init__(self, time, initial_values):
        self.time = time
        self.initial_values = initial_values
    
    def length(self):
        return self.time.t_intervals+1
    
    def dimension(self):
        return self.initial_values.shape[0]


class OdeOutput:
    
    def __init__(self, ode_input):
        self.ode_input = ode_input
        self.results = np.zeros((ode_input.length(), ode_input.dimension()))
    
    def length(self):
        return self.ode_input.length()
    
    def dimension(self):
        return self.ode_input.dimension()
    

class OdeSolver:
    
    def __init__(self, time_info, initial_values):
        ode_input = OdeInput(time_info, initial_values)
        self.ode_input = ode_input
        self.ode_output = OdeOutput(ode_input)
        
    def equation(self, t, x):
        return x
    
    def solve(self):
        dt = self.ode_input.time.dt

        # new value is set to the initial data at first
        new_value = self.ode_input.initial_values
        
        # save initial data to output
        self.ode_output.results[0] = new_value
        
        for i in range(1, self.ode_input.length()):
            # move new values to old values
            old_value = new_value
            
            time_old = self.ode_input.time.time(i - 1)
            
            k1 = self.equation(time_old, old_value)
            k2 = self.equation(time_old + 0.5 * dt, 0.5 * dt * k1 + old_value)
            k3 = self.equation(time_old + 0.5 * dt, 0.5 * dt * k2 + old_value)
            k4 = self.equation(time_old + 1.0 * dt, 1.0 * dt * k3 + old_value)
            
            new_value = old_value + (dt / 6.0) * (1.0 * k1 + 2.0 * k2 + 2.0 * k3 + 1.0 * k4)
            
            self.ode_output.results[i] = new_value
            
    def print_result(self):
        print("Solved ODE")
        print("Initial value: ", self.ode_input.initial_values)
        print("Output:")
        for i in range(0, self.ode_output.length()):
            t = self.ode_input.time.time(i)
            print("t: ", t, "y: ", self.ode_output.results[i])

