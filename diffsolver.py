import numpy as np

class odeTime:
    
    def __init__(self, tFrom, tTo, tIntervals):
        self.tFrom = tFrom
        self.tTo = tTo
        self.tIntervals = tIntervals
        
class odeInput:
    
    #input
    #    time: odeTime
    #    initialValues: numpy matrix of dimension 1*n
    
    def __init__(self, time, initialValues):
        self.time = time
        self.initialValues = initialValues
    
    def length(self):
        return self.time.tIntervals
    
    def dimension(self):
        return self.initialValues.shape[0]

class odeOutput:
    
    def __init__(self, inputForOde):
        self.inputForOde = inputForOde
        self.results = np.zeros((inputForOde.length(), inputForOde.dimension()))
    
    def length(self):
        return self.inputForOde.length()
    
    def dimension(self):
        return self.inputForOde.dimension()
    

class odeSolver:
    
    def __init__(self, timeInfo, initialValues):
        inputForOde = odeInput(timeInfo, initialValues)
        self.inputForOde = inputForOde
        self.outputForOde = odeOutput(inputForOde)
        
    def equation(self, t, x):
        return x
    
    def solve(self):
        dt = (self.inputForOde.time.tTo - self.inputForOde.time.tFrom) / float(self.inputForOde.tIntervals)
        
        oldval = np.zeros((1,self.inputForOde.dimension()))
        newval = np.zeros((1,self.inputForOde.dimension()))
        
        # save initial data to output
        self.outputForOde.results[0] = self.inputForOde.initialValues
        
        # new value is set to the initial data at first
        newval = self.outputForOde.results[0]
        
        for i in range(1, self.inputForOde.length):
            # move new values to old values
            oldval = newval
            
            tOld = self.inputForOde.time.tFrom + float(i-1) * dt
            
            k1 = equation(tOld, oldval)
            fTemp = 0.5 * dt * k1 + oldval
            
            k2 = equation(tOld + 0.5 * dt, fTemp)
            fTemp = 0.5 * dt * k2 + oldval
            
            k3 = equation(tOld + 0.5 * dt, fTemp)
            fTemp = 0.5 * dt * k3 + oldval
            
            k4= equation(tOld + 0.5 * dt, fTemp)
            fTemp = 0.5 * dt * k3 + oldval
            
            newval = oldval + (dt / 6.0) * (1.0 * k1 + 2.0 * k2 + 2.0 * k3 + 1.0 * k4)
            
            self.outputForOde.results[i] = newval
            
    def printResult(self):
        print("Solved ODE")
        print("Initial value: ", self.inputForOde.initialValues)
        print("Output:")
        dt = (self.inputForOde.time.tTo - self.inputForOde.time.tFrom) / float(self.inputForOde.tIntervals)
        for i in range(0, self.outputForOde.length):
            t = self.inputForOde.time.tFrom + dt * float(i)
            print("t: ", t, "y: ", self.outputForOde.results[i])
        