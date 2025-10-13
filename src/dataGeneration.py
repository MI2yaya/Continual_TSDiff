
import numpy as np

class SinusoidalWaves():
    def __init__(self,
        length,
        dt,
        q,
        r,
        obs_dim = 1,
        ):
        self.length = length
        self.dt = dt
        self.q = q
        self.r = r
        self.obs_dim = obs_dim
    def h_fn(self, x):
        return x

    def R_inv(self, resid):
        return resid/(self.r**2)
    
    def generate(self):
        xs=[]
        ys=[]
        amplitude=np.random.uniform(1,5)
        frequency=np.random.uniform(1,4)
        phase=np.random.uniform(0,2*np.pi)
        for step in range(self.length*(int(self.dt**-1))):
            x = np.sin(frequency*step*self.dt + phase)*amplitude + np.random.normal(0,self.q)
            xs.append(x)
            y = x + np.random.normal(0,self.r)
            ys.append(y)
        return xs, ys
            