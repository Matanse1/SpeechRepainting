import numpy as np

def load_npy_file(file_path):
    """
    Load a .npy file and return its contents.

    Parameters:
        file_path (str): Path to the .npy file.

    Returns:
        numpy.ndarray: The data loaded from the .npy file.
    """
    try:
        data = np.load(file_path)
        return data
    except Exception as e:
        print(f"Error loading .npy file: {e}")
        return None
    
if __name__ == '__main__':
    file_path = '/dsi/gannot-lab/gannot-lab1/datasets/Librispeech_mfa/phoneme-frames_filter_length=640_hop_length=160/Test/61/61-70968-0000.npy'
    data = load_npy_file(file_path)
    print(data)