import numpy as np
import torch
from .monotonic_align.core import maximum_path_c
print("Regular monotonic_align imported")

def maximum_path(value, mask):  
  """ Cython optimised version.
  value: [b, t_x, t_y]
  mask: [b, t_x, t_y]
  """
  value = value * mask
  device = value.device
  dtype = value.dtype

  # Convert to float32 (already on the device)
  value = value.to(torch.float32)

  # Create a tensor for path with int8 dtype
  path = np.zeros((value.shape[0], value.shape[1], value.shape[2]), dtype=np.int8)

  # Calculate t_x_max and t_y_max using PyTorch
  t_x_max = mask.sum(1)[:, 0].to(torch.int16)
  t_y_max = mask.sum(2)[:, 0].to(torch.int16)

  # Expand dimensions to match shape
  t_x = torch.unsqueeze(torch.unsqueeze(t_x_max, 1), 2)
  t_y = torch.unsqueeze(torch.unsqueeze(t_y_max, 1), 2)

  # Generate indices using PyTorch
  x_indices, y_indices = torch.meshgrid(torch.arange(value.shape[1], device=device), torch.arange(value.shape[2], device=device))

  # Compute condition using PyTorch operations
  condition = (x_indices >= torch.maximum(torch.tensor(0, device=device), t_x - 2 * (t_y - y_indices) + 1)) & \
              (x_indices < torch.minimum(t_x, 2 * y_indices + 1))

  # Invert condition and set corresponding value to -1e9
  condition = ~condition
  value[condition] = -1e9
  
  t_x_max = t_x_max.cpu().numpy()
  t_y_max = t_y_max.cpu().numpy()
  value = value.cpu().numpy()

  # Call the Cython function (assuming `maximum_path_c` can handle tensors)
  maximum_path_c(path, value, t_x_max, t_y_max)

  return torch.from_numpy(path).to(device=device, dtype=dtype)
