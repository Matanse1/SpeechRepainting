import torch
from just_impot_and_print import a
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))