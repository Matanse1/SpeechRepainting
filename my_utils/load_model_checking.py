import pickle
import torch

if __name__ == "__main__":
    file_path = '/dsi/gannot-lab1/users/mordehay/style-speech_weights/stylespeech.pth.tar'
    model_data = torch.load(file_path)
    print("Model data loaded successfully.")