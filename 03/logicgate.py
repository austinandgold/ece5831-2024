import numpy as np

class LogicGate():
    
    def __init__(self):
        
        pass

    def andgate (self,x1,x2):
        w = np.array([1, 1, 1])          
        b = -1.9
        X = np.array([x1, x2, b])

        y = np.sum(w*X)

        if y > 0:
            return 1
        else:
            return 0
        
    def orgate (self,x1,x2):
        w = np.array([1, 1, 1])          
        b = -0.9
        X = np.array([x1, x2, b])

        y = np.sum(w*X)

        if y > 0:
            return 1
        else:
            return 0

    def nandgate (self,x1,x2):
        w = np.array([-1, -1, 1])          
        b = 1.9
        X = np.array([x1, x2, b])

        y = np.sum(w*X)

        if y > 0:
            return 1
        else:
            return 0
        
    def norgate (self,x1,x2):
        w = np.array([-1, -1, 1])          
        b = 0.9
        X = np.array([x1, x2, b])

        y = np.sum(w*X)

        if y > 0:
            return 1
        else:
            return 0
        
    def xorgate (self,x1,x2):
        w = np.array([0.5, 0.5, 1])          
        b = -0.5
        X = np.array([x1, x2, b])

        y = np.sum(w*X)

        if y == 0:
            return 1
        else:
            return 0