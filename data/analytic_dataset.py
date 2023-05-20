from torch.utils.data import Dataset
from math_utils.logistic_coefficient_map import logistic_taylor_orbit, rescale, tau_from_last_coef
import numpy as np

NUM_COEFFICIENTS = 100

class AnalyticDataset(Dataset):
    def __init__(self, N, transform=None) -> None:
        super().__init__()
        self.N = N
        self.transform = transform
        self.initial_values = np.random.uniform(low=-1, high=1, size=N)
        self.taus = np.ones_like(self.initial_values)
    
    def __len__(self):
        return self.N

    def __getitem__(self, index):
        iv = np.array([self.initial_values[index]])
        tau = self.taus[index]
        coefficients = logistic_taylor_orbit(self.initial_values[index], NUM_COEFFICIENTS, tau=tau)
        if tau == 1:
            tau = tau_from_last_coef(coefficients)
            coefficients = rescale(coefficients, tau)
            self.taus[index] = tau
        
        if self.transform is not None:
            iv = self.transform(iv)
            coefficients = self.transform(coefficients)
        return iv, coefficients
