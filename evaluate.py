import torch
from models.almost_fully_convolutional import FullyConvolutionLogistic


ckpt = torch.load('best.ckpt')
model = FullyConvolutionLogistic(100)

model.load_state_dict(ckpt['model_state_dict'])

dataset = AnalyticDataset(1000, transform=transform)

i, o = dataset[1]

model(i.unsqueeze(0))