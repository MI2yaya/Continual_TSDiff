
import numpy as np
import torch
def make_kf_matrices_for_sinusoid(generator, past_obs=None, mode="const_vel"):
    """
    generator: SinusoidalWaves instance (has .dt, .q, .r)
    past_obs: optional 1D numpy array of past observations for frequency estimation
    mode: "const_vel" or "osc"
    Returns: A, H, Q, R (numpy arrays)
    """
    dt = generator.dt
    q = float(generator.q)
    r = float(generator.r)

    if mode == "const_vel":
        A = np.array([[1.0, dt],
                      [0.0, 1.0]], dtype=float)
        H = np.array([[1.0, 0.0]], dtype=float)
        # acceleration spectral density proxy
        q_c = q**2
        Q = q_c * np.array([[dt**3/3.0, dt**2/2.0],
                            [dt**2/2.0, dt]], dtype=float)
        R = np.array([[r**2]], dtype=float)
        return A, H, Q, R

    elif mode == "osc":
        # estimate dominant frequency via FFT
        x = np.asarray(past_obs).astype(float).flatten()
        x = x - x.mean()
        n = len(x)
        freqs = np.fft.rfftfreq(n, dt)  # cycles per second
        X = np.fft.rfft(x)
        idx = np.argmax(np.abs(X))
        f_peak = freqs[idx]
        # protect against zero freq
        f_peak = max(f_peak, 1e-6)
        omega = 2 * np.pi * f_peak

        # oscillator A
        A = np.array([[np.cos(omega*dt), (1.0/omega)*np.sin(omega*dt)],
                      [-omega*np.sin(omega*dt), np.cos(omega*dt)]], dtype=float)
        H = np.array([[1.0, 0.0]], dtype=float)
        Q = 0.01 * np.eye(2, dtype=float)   # small process noise; tune if needed
        R = np.array([[r**2]], dtype=float)
        return A, H, Q, R

    else:
        raise ValueError("mode must be 'const_vel' or 'osc'")

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
        eps = 1e-6
        var = (self.r ** 2) + eps
        R_inv = resid / var
        R_inv = R_inv / (R_inv.std(dim=1, keepdim=True) + 1e-5)
        return R_inv
    
    
    def generate(self):
        xs=[]
        ys=[]
        amplitude=np.random.uniform(1,5.0)
        frequency=np.random.uniform(1,5.0)
        phase=np.random.uniform(0,2*np.pi)
        for step in range(self.length*(int(self.dt**-1))):
            x = np.sin(frequency*step*self.dt + phase)*amplitude + np.random.normal(0,self.q)
            xs.append(x)
            y =  x + np.random.normal(0,self.r)
            ys.append(y)
        return xs, ys
    
    
    
class FourthOrderRungeKutta():
    def __init__(self,
        length,
        dt,
        q,
        r,
        obs_dim = 1,
        ):
        self.length = length
        self.dt = dt
        self.r = r
    def f(self,z):
        z1_dot = 10*(z[1] - z[0])
        z2_dot = z[0] * (28 - z[2]) - z[1]
        z3_dot = z[0] * z[1] - (8/3) * z[2]
        return np.array([z1_dot,z2_dot,z3_dot])
    
    def h_fn(self,x):
        return 0.5 * (x[..., 0]**2 + x[..., 1]**2) + 0.7 * x[..., 2:3] 
    
    def R_inv(self,resid):
        eps = 1e-6
        var = (self.r ** 2) + eps
        R_inv = resid / var
        R_inv = R_inv / (R_inv.std(dim=1, keepdim=True) + 1e-5)
        return R_inv
    
    def RK4_step(self,z):
        k1 = self.f(z)
        k2 = self.f(z + (self.dt/2)*k1)
        k3 = self.f(z + (self.dt/2)*k2)
        k4 = self.f(z + self.dt*k3)
        return z + (self.dt / 6) * (k1+2*k2+2*k3+k4)
    
    def nextUpdate(self,z):
        z = self.RK4_step(z)
        noise = np.random.multivariate_normal(mean=np.zeros(3), cov=(0.02**2) * np.eye(3))
        return z + noise
    
    def measurement(self,z):
        y = .5 * (z[0]**2 + z[1]**2) + .7*z[2]
        noise = np.random.normal(0,self.r)
        return y + noise
    
    def generate(self):
        z = np.zeros(3)
        xs=[]
        ys=[]
        num_steps = int(self.length / self.dt)
        for _ in range(num_steps):
            z = self.nextUpdate(z)
            xs.append(z)
            ys.append(self.measurement(z))
        return xs,ys