from models.almost_fully_convolutional import FullyConvolutionLogistic
import torch
from torch import nn
from torchvision.transforms import transforms
from data.analytic_dataset import AnalyticDataset
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np

DEVICE_ID = 'cuda'
EPOCHS = 200
LR = .0005

transform = transforms.Compose(
    [
        transforms.Lambda(
            lambd=lambda x: x.astype(np.float32)
        ),
        transforms.Lambda(
            lambd=lambda x: torch.from_numpy(x)
        ),
    ]
)

dataset = AnalyticDataset(10000, transform=transform)

lengths = (9000, 1000)
trainset, valset = random_split(dataset, lengths)

trainloader = DataLoader(trainset, batch_size=100, num_workers=6)
valloader = DataLoader(valset, batch_size=1000)

model = FullyConvolutionLogistic(100)

model.to(DEVICE_ID)
optim = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.L1Loss()
# loss_fn = nn.L1Loss(reduction='none')
# weights = torch.exp(-torch.arange(0, 100)).to(DEVICE_ID)

best_loss = np.Inf
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for in_batch, out_batch in tqdm(trainloader):
        in_batch = in_batch.to(DEVICE_ID)
        out_batch = out_batch.to(DEVICE_ID)
        optim.zero_grad()

        pred = model(in_batch)
        # loss = (loss_fn(pred, out_batch) * weights).sum()
        loss = loss_fn(pred, out_batch)
        loss.backward()
        optim.step()
        total_loss += loss
    print(f'Epoch {epoch} | Loss {total_loss}')
    torch.save(
        {
            'epoch': epoch,
            'loss': total_loss,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optim.state_dict()
        },
        'latest.ckpt',
    )
    if total_loss < best_loss:
        torch.save(
            {
                'epoch': epoch,
                'loss': total_loss,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optim.state_dict()
            },
            'best.ckpt',
        )



