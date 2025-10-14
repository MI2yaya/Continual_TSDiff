
import numpy as np

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
        if past_obs is None or len(past_obs) < 8:
            # fallback
            return make_kf_matrices_for_sinusoid(generator, past_obs, mode="const_vel")

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
        eps = 1e+3
        return resid/((self.r)**2+eps)
    
    def generate(self):
        xs=[]
        ys=[]
        amplitude=1
        frequency=np.random.uniform(1,4)
        phase=np.random.uniform(0,2*np.pi)
        for step in range(self.length*(int(self.dt**-1))):
            x = np.sin(frequency*step*self.dt + phase)*amplitude + np.random.normal(0,self.q)
            xs.append(x)
            y = x + np.random.normal(0,self.r)
            ys.append(y)
        return xs, ys
            