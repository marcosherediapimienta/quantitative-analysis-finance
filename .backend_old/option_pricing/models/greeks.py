class Greeks:
    def __init__(self, delta: float, gamma: float, vega: float, theta: float, rho: float):
        self.delta = delta
        self.gamma = gamma
        self.vega = vega
        self.theta = theta
        self.rho = rho

    def as_dict(self):
        return {
            'delta': self.delta,
            'gamma': self.gamma,
            'vega': self.vega,
            'theta': self.theta,
            'rho': self.rho
        }

    def __repr__(self):
        return (f"Greeks(delta={self.delta}, gamma={self.gamma}, vega={self.vega}, theta={self.theta}, rho={self.rho})")
