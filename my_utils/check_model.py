import torch
import torch.nn as nn

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
print(torch.backends.cudnn.enabled)
# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Conv1d(10, 10, 3)

    def forward(self, x):
        return self.fc(x)

input_data = torch.randn(1, 10, 16000*3, dtype=torch.float32).to(device)
model = SimpleModel().to(device)
model = torch.compile(model)

output_data = model(input_data)
print(output_data)
