
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'  # Only GPU 1 is visible

import torch
# device = torch.device("cuda:0")  # 'cuda:0' now refers to GPU 1
# model.to(device)
print(torch.cuda.device_count())  # Should return 1
print(torch.cuda.current_device())  # Should return 0 (but it's actually GPU 1)
print(torch.cuda.get_device_name(0))  # Should print GPU 1's name
x = torch.zeros(5000, 5000).cuda()
while True:
    x = x @ x