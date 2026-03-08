import torch
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import librosa
import re
import json
from string import punctuation
from g2p_en import G2p
import soundfile as sf
from models.StyleSpeech import StyleSpeech
from text import text_to_sequence
import audio as Audio
import utils
from mel2wav import MelVocoder
#For HiFi-GAN

from hifi_gan import utils as vocoder_utils
from hifi_gan.env import AttrDict
from hifi_gan.generator import Generator as Vocoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_StyleSpeech(config, checkpoint_path):
    model = StyleSpeech(config).to(device=device)
    state_dict = torch.load(checkpoint_path)['model']
    keys_to_remove = ['encoder.position_enc', 'variance_adaptor.length_regulator.position_enc', 'decoder.position_enc'] # in my case the audio can be much longer than 1000 frames, so i dont wwant each batch the position encoding to be created again
    for key in keys_to_remove:
        if key in state_dict:
            del state_dict[key]
    model.load_state_dict(state_dict, strict=False)
    # model.load_state_dict(torch.load(checkpoint_path)['model'], strict=False)
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters: {num_params}')
    return model


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, lexicon_path, with_space):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(lexicon_path)

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        elif w == " ":
            if with_space:
                phones += ["sp"]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(text_to_sequence(phones, ['english_cleaners']))
    # value_space = np.array(text_to_sequence('{sp}', ['english_cleaners']))
    # sequence =  np.pad(sequence, pad_width=1, mode='constant', constant_values=value_space)
    return torch.from_numpy(sequence).to(device=device)


def preprocess_audio(audio_file, _stft):
    wav, sample_rate = librosa.load(audio_file, sr=None)
    if sample_rate != 16000:
        wav = librosa.resample(wav, sample_rate, 16000)
    mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, _stft)
    return torch.from_numpy(mel_spectrogram).to(device=device)


# def get_StyleSpeech(config, checkpoint_path):
#     model = StyleSpeech(config).to(device=device)
#     model.load_state_dict(torch.load(checkpoint_path)['model'])
#     model.eval()
#     num_params = sum(p.numel() for p in model.parameters())
#     print(f'Total number of parameters: {num_params}')
#     return model


def synthesize(args, model, _stft):   
    # preprocess audio and text
    ref_mel = preprocess_audio(args.ref_audio, _stft).transpose(0,1).unsqueeze(0)
    src = preprocess_english(args.text, args.lexicon_path, with_space=args.with_space).unsqueeze(0)
    src_len = torch.from_numpy(np.array([src.shape[1]])).to(device=device)
    
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Extract style vector
    style_vector = model.get_style_vector(ref_mel)

    # Forward
    masked_frame_number = ref_mel.shape[-2]
    outputs = model.inference(style_vector, src, src_len, masked_frame_number=None)
    mel_output, src_embedded, d_prediction, p_prediction, e_prediction, src_mask, mel_mask, mel_len = outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6], outputs[7]
    print(torch.exp(d_prediction))
    rounded_duration =  torch.clamp(torch.round(torch.exp(d_prediction)-1.0), min=0)    
    print(f"Duraion prediction: {rounded_duration}")
    mel_ref_ = ref_mel.cpu().squeeze().transpose(0, 1).detach()
    mel_ = mel_output.cpu().squeeze().transpose(0, 1).detach()

    # plotting
    utils.plot_data([mel_ref_.numpy(), mel_.numpy()], 
        ['Ref Spectrogram', 'Synthesized Spectrogram'], filename=os.path.join(save_path, 'plot.png'))
    if args.with_space:
        print('Load HiFi-GAN')
        config_file = 'hifi_gan/config.json'
        with open(config_file) as f:
            data = f.read()
        json_config = json.loads(data)
        h = AttrDict(json_config)
        vocoder = Vocoder(h).cuda()
        checkpoint_file = '/dsi/gannot-lab1/users/mordehay/hifi_gan/g_02400000'
        state_dict_g = vocoder_utils.load_checkpoint(checkpoint_file, 'cuda')
        vocoder.load_state_dict(state_dict_g['generator'])
        vocoder.eval()
        vocoder.remove_weight_norm()
        print('Finish Loading HiFi-GAN')
        wav = vocoder(mel_output.transpose(-2, -1)).squeeze().cpu().detach().numpy()
        sf.write(os.path.join(save_path, 'synthesized.wav'), wav, 16000)
    else:
        vocoder = MelVocoder(path='/dsi/gannot-lab1/users/mordehay/melgen-stylespeech-vocoder/')
        wav = vocoder.inverse(mel_output.transpose(-2, -1)).squeeze().cpu().detach().numpy()
        sf.write(os.path.join(save_path, 'synthesized.wav'), wav, 16000)
    print('Generate done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    my_style_speech = True
    if my_style_speech:
        parser.add_argument("--checkpoint_path", type=str, default='/dsi/gannot-lab1/users/mordehay/my_StyleSpeech/with-space_masked-mel4style-vec/ckpt/checkpoint_205000.pth.tar',
            help="Path to the pretrained model")
        parser.add_argument('--config', default='/dsi/gannot-lab1/users/mordehay/my_StyleSpeech/with-space/config.json')
        parser.add_argument("--with_space", default=True, help="Use space token in phoneme sequence")
    else:
        parser.add_argument("--checkpoint_path", type=str, default='/dsi/gannot-lab1/users/mordehay/style-speech_weights/stylespeech.pth.tar',
            help="Path to the pretrained model")
        parser.add_argument('--config', default='/dsi/gannot-lab1/users/mordehay/style-speech_weights/config.json')
        parser.add_argument("--with_space", default=False, help="Use space token in phoneme sequence")
    

    
    parser.add_argument("--save_path", type=str, default='/home/dsi/moradim/SpeechRepainting/StyleSpeech/results_176500_sample=69_masked-ref_masked-model_again/')
    parser.add_argument("--ref_audio", type=str, default='/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/DiT_Anechoic_LibSp_conditional-my-tts-melspec_positional_emd=InputEmbedding1_cross-custom-attn-noise_w-masked-pix=0.5/dit-net_dim768_depth9_heads12_dim-head64_dropout0.1_ff_mult2_T400_betaT0.02/repeat_all_freq-length=100_skip=150_cp=112000_mel_text=True_withoutLM_phoneme-with-space/w1=2_w2=0.8_asr_start=320_mask=True/sample_69/masked_audio_time.wav',
        help="path to an reference speech audio sample")

    parser.add_argument("--text", type=str, default='IN THE MODERN WELL CONSTRUCTED PLAY HE SIMPLY RINGS UP AN IMAGINARY CONFEDERATE AND TELLS HIM WHAT HE IS GOING TO DO COULD ANYTHING BE MORE NATURAL',
        help="raw text to synthesize")
    parser.add_argument("--lexicon_path", type=str, default='/home/dsi/moradim/SpeechRepainting/StyleSpeech/lexicon/librispeech-lexicon.txt')
    args = parser.parse_args()
    
    with open(args.config) as f:
        data = f.read()
    json_config = json.loads(data)
    config = utils.AttrDict(json_config)

    # Get model
    model = get_StyleSpeech(config, args.checkpoint_path)
    print('model is prepared')
    if args.with_space:
        log = 'log'
    else:
        log = 'log10'
    _stft = Audio.stft.TacotronSTFT(
                config.filter_length,
                config.hop_length,
                config.win_length,
                config.n_mel_channels,
                config.sampling_rate,
                config.mel_fmin,
                config.mel_fmax,
                log=log)

    # Synthesize
    synthesize(args, model, _stft)
