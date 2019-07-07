
#%%
from Data_Management import CRNDataset
import torch


#%%
data_set = CRNDataset(
    max_input_height_width=(32, 64),
    root="C:\\CityScapes Samples\\Train\\",
    split="train",
    num_classes=35
)


#%%
torch.arange(0, 6).view(3,2) % 3


#%%
a, b = data_set[0]


#%%
b[0]
