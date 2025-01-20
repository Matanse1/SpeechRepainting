import sys
sys.path.append("/home/dsi/moradim/SpeechRepainting")
from models.utils import get_phones_dict


phoneme_dict_path = "/home/dsi/moradim/SpeechRepainting/phones.txt"
phoneme_dict_p2d, _ = get_phones_dict(phoneme_dict_path)
print(phoneme_dict_p2d)
