

import torch
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

# def mask_with_shape_mask(mel_spec, number):
#     """
#     Creates a binary mask for the mel spectrogram based on a drawn shape.
#     shape_function should draw on an Image object to create the mask.
#     """
#     img = Image.new("L", (mel_spec.shape[0], mel_spec.shape[1]), 255)
#     draw = ImageDraw.Draw(img)
#     width, height = img.size
#     draw.text((width // 3, height // 4), str(number), fill=0, anchor="ms")
#     shape_mask = (np.array(img) < 128).astype(np.float32)  # Binary mask: 0 for mask, 1 otherwise
#     mask = torch.tensor(shape_mask).T
#     noise = torch.randn_like(mel_spec)
#     return mel_spec * mask + noise * (1 - mask), mask
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def mask_with_number_two(mel_spec):
    """
    Creates a binary mask for the mel spectrogram in the shape of number 2.
    The number will be flipped over the Y-axis.
    """
    freq_dim, time_dim = mel_spec.shape
    
    # Make the figure wider by increasing the width ratio
    fig, ax = plt.subplots(figsize=(time_dim/5, freq_dim/10))  # Doubled the width ratio
    
    font_size = min(freq_dim, time_dim) * 5
    
    ax.text(0.5, 0.5, "2", 
            color="black",
            fontsize=font_size,
            ha="center",
            va="center",
            alpha=1.0,
            fontweight='bold',
            fontfamily='serif')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    img = Image.fromarray(img).convert("L")
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = img.resize((time_dim, freq_dim), Image.Resampling.LANCZOS)
    
    shape_mask = (np.array(img) < 128).astype(np.float32)
    mask = torch.tensor(shape_mask)
    mask = mask.flip((0, 1))
    
    noise = torch.randn_like(mel_spec)
    masked_spec = mel_spec * mask + noise * (1 - mask)
    
    return masked_spec, mask



mel_spec = torch.rand(80, 600)  # Example mel spectrogram (80 frequencies, 100 time steps)

# Create the binary mask
print("start mask")
# masked_mel_spec, mask = mask_with_number_two(mel_spec, "2")
masked_mel_spec, mask = mask_with_number_two(mel_spec)
print("finish mask")


# Save the masked mel spectrogram as an image
plt.imshow(mask.numpy(), aspect='auto', origin='lower')
plt.colorbar(format='%+2.0f dB')
plt.title('Masked Mel Spectrogram')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.savefig('/home/dsi/moradim/SpeechRepainting/masked_mel_spec.png')
plt.close()