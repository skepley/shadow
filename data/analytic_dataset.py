from torch.utils.data import Dataset
from math_utils.logistic_coefficient_map import logistic_taylor_orbit
import numpy as np

NUM_COEFFICIENTS = 100

class AnalyticDataset(Dataset):
    def __init__(self, N, transform=None) -> None:
        super().__init__()
        self.N = N
        self.transform = transform
        self.initial_values = np.random.uniform(low=-1, high=1, size=N)
    
    def __len__(self):
        return self.N

    def __getitem__(self, index):
        iv = np.array([self.initial_values[index]])
        coefficients = logistic_taylor_orbit(self.initial_values[index], NUM_COEFFICIENTS)
        if self.transform is not None:
            iv = self.transform(iv)
            coefficients = self.transform(coefficients)
        return iv, coefficients
