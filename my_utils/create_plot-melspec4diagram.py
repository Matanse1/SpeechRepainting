import sys
sys.path.append('/home/dsi/moradim/SpeechRepainting/')
import torch
import os
import matplotlib.image

path2mel = '/home/dsi/moradim/SpeechRepainting/files4plots/4diagram_noisy-melspec-t=50.pt'

melspec = torch.load(path2mel)
melspec = melspec[0].squeeze().cpu().numpy()
save_path = path2mel.replace('.pt', '.png')
matplotlib.image.imsave(save_path, melspec[::-1])