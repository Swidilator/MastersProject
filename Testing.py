
#%%
from Data_Management import CRNDataset
import torch
from Perceptual_Loss import PerceptualLossNetwork
from CRN import CRN


#%%
data_set = CRNDataset(
    max_input_height_width=(32, 64),
    root="C:\\CityScapes Samples\\Train\\",
    split="train",
    num_classes=35
)


#%%
a = torch.arange(0, 280).view(35, 2, 4)
a1 = torch.zeros((35, 4, 8)).float()
a1[:35, :2, :4] = a
b = torch.arange(0, 1120).view(35, 4, 8).float()

c = torch.cat((a1, b))

#%%
a, b = data_set[0]


#%%
b[0]
# %%
pln = PerceptualLossNetwork()

# %%
pln(a[0].unsqueeze(0).float(), a[0].unsqueeze(0).float())

# %%
crn = CRN(35)

# %%
out = crn(b.unsqueeze())

# %%
data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
    data_set, batch_size=1, shuffle=True, num_workers=1
)

# %%
image, smnt = next(iter(data_loader))

# %%
