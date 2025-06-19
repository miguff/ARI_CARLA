import numpy as np

class OUActionNoise():
    """
    Ornstein-Uhlenbeck process for generating noise, commonly used in continuous 
    action spaces in reinforcement learning to promote exploration.

    Attributes:
        mu (np.ndarray): The long-running mean (drift term).
        sigma (float): The volatility parameter (standard deviation of the noise).
        theta (float): The speed of mean reversion.
        dt (float): The time step size.
        x0 (np.ndarray or None): The initial state of the process.
        x_prev (np.ndarray): The previous state (used internally).

    Methods:
        __call__():
            Generates the next noise value based on the OU process.
        
        reset():
            Resets the internal state to the initial value or zeros.
    """
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0 =None):
        """
        Initializes the OUActionNoise instance.

        Parameters:
            mu (np.ndarray): The mean to revert to over time.
            sigma (float, optional): The scale of the noise. Default is 0.15.
            theta (float, optional): The rate of mean reversion. Default is 0.2.
            dt (float, optional): Time step size. Default is 1e-2.
            x0 (np.ndarray or None, optional): Initial state. If None, defaults to zeros.
        """
        self.theta = theta
        self.mean = mu
        self.std = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        """
        Computes the next value in the OU process and updates the internal state.

        Returns:
            np.ndarray: The new noise value.
        """
        x = self.x_prev + self.theta * (self.mean - self.x_prev) * self.dt + \
                self.std * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        self.x_prev = x

        return x

    def reset(self):
        """
        Resets the internal state to the initial state (x0) or to zero if x0 is not provided.
        """
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mean)