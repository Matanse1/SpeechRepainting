
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality as PESQ
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility as STOI
import librosa
import numpy as np
import sys
sys.path.append('/home/dsi/moradim/SpeechRepainting')
import torchaudio
from PLCMOS.plc_mos import PLCMOSEstimator
import whisper
import torch
import jiwer
from whisper.normalizers import EnglishTextNormalizer
import glob
import os
import csv
import pandas as pd
from tqdm import tqdm
import cProfile
from pathlib import Path


import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf

def is_number(value):
    return isinstance(value, (int, float, np.number))


INPUT_LENGTH = 9.01
"""
class ComputeScoreDNSMOS:
    def __init__(self, primary_model_path, p808_model_path) -> None:
        self.onnx_sess = ort.InferenceSession(primary_model_path)
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path)
    
    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size+1, hop_length=hop_length, n_mels=n_mels)
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max)+40)/40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021,  0.005101  ,  1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499,  0.44276479, -0.1644611 ,  0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
            p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
            p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def compute_full_audio_score(self, audio_signal, sampling_rate=16000, is_personalized_MOS=False):
        input_features = np.array(audio_signal).astype('float32')[np.newaxis,:]
        p808_input_features = np.array(self.audio_melspec(audio=audio_signal)).astype('float32')[np.newaxis, :, :]
        oi = {'input_1': input_features}
        p808_oi = {'input_1': p808_input_features}
        
        p808_mos = 0 #self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
        mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
        mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(mos_sig_raw, mos_bak_raw, mos_ovr_raw, is_personalized_MOS)
        
        return {
            'OVRL_raw': mos_ovr_raw,
            'SIG_raw': mos_sig_raw,
            'BAK_raw': mos_bak_raw,
            'OVRL': mos_ovr,
            'SIG': mos_sig,
            'BAK': mos_bak,
            'P808_MOS': p808_mos
        }
"""

class ComputeScoreDNSMOS:
    def __init__(self, primary_model_path, p808_model_path) -> None:
        self.onnx_sess = ort.InferenceSession(primary_model_path)
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path)
    
    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size+1, hop_length=hop_length, n_mels=n_mels)
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max)+40)/40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021,  0.005101  ,  1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499,  0.44276479, -0.1644611 ,  0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
            p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
            p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])

        return p_sig(sig), p_bak(bak), p_ovr(ovr)

    def compute_full_audio_score(self, audio_signal, sampling_rate=16000, is_personalized_MOS=False):
        fs = sampling_rate
        len_samples = int(INPUT_LENGTH * fs) 
        
        while len(audio_signal) < len_samples:
            audio_signal = np.append(audio_signal, audio_signal)
        
        num_hops = int(np.floor(len(audio_signal)/fs) - INPUT_LENGTH) + 1
        hop_len_samples = fs
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio_signal[int(idx*hop_len_samples) : int((idx+INPUT_LENGTH)*hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype('float32')[np.newaxis,:]
            p808_input_features = np.array(self.audio_melspec(audio=audio_seg[:-160])).astype('float32')[np.newaxis, :, :]
            oi = {'input_1': input_features}
            p808_oi = {'input_1': p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig, mos_bak, mos_ovr = self.get_polyfit_val(mos_sig_raw, mos_bak_raw, mos_ovr_raw, is_personalized_MOS)
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)

        return {

            'OVRL_raw': np.mean(predicted_mos_ovr_seg_raw),
            'SIG_raw': np.mean(predicted_mos_sig_seg_raw),
            'BAK_raw': np.mean(predicted_mos_bak_seg_raw),
            'OVRL': np.mean(predicted_mos_ovr_seg),
            'SIG': np.mean(predicted_mos_sig_seg),
            'BAK': np.mean(predicted_mos_bak_seg),
            'P808_MOS': np.mean(predicted_p808_mos)
        }
        
        
def count_directories(path):
    return len(glob.glob(f"{path}/*/"))

def get_power(x, nfft):
    S = librosa.stft(x, n_fft=nfft)
    S = np.log(np.abs(S) ** 2 + 1e-8)
    return S


def LSD(x_hr, x_pr):
    S1 = get_power(x_hr, nfft=2048)
    S2 = get_power(x_pr, nfft=2048)
    lsd = np.mean(np.sqrt(np.mean((S1 - S2) ** 2 + 1e-8, axis=-1)), axis=0)
    S1 = S1[-(len(S1) - 1) // 2:, :]
    S2 = S2[-(len(S2) - 1) // 2:, :]
    lsd_high = np.mean(np.sqrt(np.mean((S1 - S2) ** 2 + 1e-8, axis=-1)), axis=0)
    return lsd, lsd_high

class Metrics:
    def __init__(self, sampling_rate=16000, whisper_model_name= 'medium.en', path2whisper_model='/dsi/gannot-lab1/users/mordehay/whisper_models'):
        self.sampling_rate = sampling_rate
        self.stoi = STOI(sampling_rate)
        self.pesq = PESQ(sampling_rate, 'wb') # wide band signal (speech)
        self.plcmos = PLCMOSEstimator()
        self.whisper_model = whisper.load_model(whisper_model_name, download_root=path2whisper_model) # choices=['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large'], default='medium.en', help='Pretrained ASR model. Those with .en are english-only')
        print(
            f"Model is {'multilingual' if self.whisper_model.is_multilingual else 'English-only'} "
            f"and has {sum(np.prod(p.shape) for p in self.whisper_model.parameters()):,} parameters."
        )
        self.options = whisper.DecodingOptions(language="en", without_timestamps=True)
        self.normalizer = EnglishTextNormalizer()
        self.dnsmos = ComputeScoreDNSMOS(primary_model_path='/home/dsi/moradim/SpeechRepainting/DNSMOS/DNSMOS/sig_bak_ovr.onnx', p808_model_path='/home/dsi/moradim/SpeechRepainting/DNSMOS/DNSMOS/model_v8.onnx')
        
    def calc_dnsmos(self, audio):
        return self.dnsmos.compute_full_audio_score(audio.cpu().numpy())
    def calc_wer(self, target_text, wav):
        waveform = whisper.pad_or_trim(wav.flatten()).to(wav.device)
        mel = whisper.log_mel_spectrogram(waveform)
        hypo = self.whisper_model.decode(mel, self.options).text
        hypo_clean = self.normalizer(hypo)
        reference_clean = self.normalizer(target_text)
        wer = jiwer.wer(reference_clean, hypo_clean)
        num_words_ref = len(reference_clean.split())
        wrong_words = int(wer * num_words_ref)
        wer = wer * 100
        return round(wer, 4), num_words_ref, wrong_words, reference_clean, hypo_clean
    def calc_wer_2_texts(self, target_text, pred_text):
        hypo_clean = self.normalizer(pred_text)
        reference_clean = self.normalizer(target_text)
        wer = jiwer.wer(reference_clean, hypo_clean)
        num_words_ref = len(reference_clean.split())
        wrong_words = int(wer * num_words_ref)
        wer = wer * 100
        return round(wer, 4)
    
    def compute_metrics(self, pred, tar_wav, target_text):
        if len(pred) < len(tar_wav):
            pred = torch.cat([pred, torch.zeros(len(tar_wav) - len(pred)).to(pred.device)])
        else:
            tar_wav = torch.cat([tar_wav, torch.zeros(len(pred) - len(tar_wav)).to(tar_wav.device)])

        wer, num_words_ref, wrong_words, reference_clean, hypo_clean = self.calc_wer(target_text, pred)

        stoi = self.stoi(pred, tar_wav).item()

        lsd, _ = LSD(tar_wav.cpu().numpy(), pred.cpu().numpy())
        
        plcmos_pred = self.plcmos.run(pred.cpu().numpy(), self.sampling_rate)

        pesq = self.pesq(torch.tensor(pred), torch.tensor(tar_wav)).item()

        metrics = {
            'WER': round(wer, 3),
            'trans': hypo_clean,
            'num_wrong_words': wrong_words,
            'total_num_words': num_words_ref,
            "plcmos_pred": round(plcmos_pred, 3),
            'LSD': round(lsd, 3),
            'STOI': round(stoi, 3),
            'PESQ': round(pesq, 3),
                }
        return metrics

    
    def compute_init_metrics(self, masked_audio, tar_wav):
        if len(masked_audio) < len(tar_wav):
            masked_audio = torch.cat([masked_audio, torch.zeros(len(tar_wav) - len(masked_audio)).to(masked_audio.device)])
        else:
            tar_wav = torch.cat([tar_wav, torch.zeros(len(masked_audio) - len(tar_wav)).to(tar_wav.device)])

        stoi_init = self.stoi(masked_audio, tar_wav).item()
        
        lsd_init, _ = LSD(tar_wav.cpu().numpy(), masked_audio.cpu().numpy())

        plcmos_tar = self.plcmos.run(tar_wav.cpu().numpy(), self.sampling_rate)
        
        pesq_init = self.pesq(torch.tensor(masked_audio), torch.tensor(tar_wav)).item()
        
        metrics = {
            "plcmos_target_init": round(plcmos_tar, 3),
            'LSD_init': round(lsd_init, 3),
            'STOI_init': round(stoi_init, 3),
            'PESQ_init': round(pesq_init, 3),
                }
        return metrics

    def calc_plcmos(self, audio):
        return self.plcmos.run(audio.cpu().numpy(), self.sampling_rate)


def main(pathes2data, name_csv='dit', mel_text_bool=True):
    # compute_metrics = Metrics()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # masked_audio, sr = torchaudio.load('/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/LibSp_wavlm-base-plus-rep_w_masked_pix=0.8_two_branch=True_specific_hidden_states_randn-filled/wnet_h512_d12_T400_betaT0.02/small_gap_mask/w1=2_w2=1.5_asr_start=270_mask=True/sample_0/masked_audio_time.wav')
    # pred, sr = torchaudio.load('/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/LibSp_wavlm-base-plus-rep_w_masked_pix=0.8_two_branch=True_specific_hidden_states_randn-filled/wnet_h512_d12_T400_betaT0.02/small_gap_mask/w1=2_w2=1.5_asr_start=270_mask=True/sample_0/generated_audio_hifi_gan.wav')
    # tar_wav, sr = torchaudio.load('/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/LibSp_wavlm-base-plus-rep_w_masked_pix=0.8_two_branch=True_specific_hidden_states_randn-filled/wnet_h512_d12_T400_betaT0.02/small_gap_mask/w1=2_w2=1.5_asr_start=270_mask=True/sample_0/gt_audio_hifi_gan.wav')
    # pred = pred[0].to(device)
    # masked_audio = masked_audio[0].to(device)
    # tar_wav = tar_wav[0].to(device)
    # target_text = "Hello world"
    # metrics = compute_metrics.compute_metrics(masked_audio, pred, tar_wav, target_text)
    # print(metrics)
    
    # path2data = '/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/LibSp_wavlm-base-plus-rep_w_masked_pix=0.8_two_branch=True_specific_hidden_states_randn-filled/wnet_h512_d12_T400_betaT0.02/small_gap_mask/w1=2_w2=1.5_asr_start=270_mask=True'
    # pathes2data = ['/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/Unet_Anechoic_LibSp_wavlm-conditional_w-masked-pix=0.8/unet_dim64_dim_mults1_2_4_T400_betaT0.02/as-train-gap_cp=732000_mel_text=True_withoutLM/w1=2_w2=0.8_asr_start=320_mask=True',
                #    '/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/Unet_Anechoic_LibSp_wavlm-conditional_w-masked-pix=0.8/unet_dim64_dim_mults1_2_4_T400_betaT0.02/medium-gap_cp=732000_mel_text=True_withoutLM/w1=2_w2=0.8_asr_start=320_mask=True']
    # pathes2data  =    ['/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/Unet_Anechoic_LibSp_wavlm-conditional_w-masked-pix=0.8/unet_dim64_dim_mults1_2_4_T400_betaT0.02/small-gap_cp=732000_mel_text=True_withoutLM/w1=2_w2=0.8_asr_start=320_mask=True',
                # '/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/Unet_Anechoic_LibSp_unconditional/unet_dim64_dim_mults1_2_4_T400_betaT0.02/as-train-gap_cp=1256000_mel_text=True_withoutLM/w1=2_w2=0.8_asr_start=320_mask=True']#,
    # combine_dataframes = False
    # pathes2data = ['/home/dsi/moradim/SpeechRepainting/StyleSpeech/results_greater_7_gap=100',
    #                '/home/dsi/moradim/SpeechRepainting/StyleSpeech/results_greater_7_gap=50',
    #                '/home/dsi/moradim/SpeechRepainting/StyleSpeech/results_greater_7_gap=25']
    save_csv_path = f"/home/dsi/moradim/SpeechRepainting/{name_csv}_metric_results.csv"
        # Ask if user wants to remove the existing CSV file
    user_input = input(f"Do you want to remove the existing file at {save_csv_path}? (yes/no): ")

    if user_input.lower() == 'yes':
        # Remove the CSV file
        csv_path = Path(save_csv_path)
        if csv_path.exists():
            csv_path.unlink()  # Remove the file
            print(f"File {save_csv_path} has been removed.")
        else:
            print(f"File {save_csv_path} does not exist.")
    else:
        print(f"Proceeding without removing the file.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocoder_names = ["bigvgan", "hifi_gan"] 
    # vocoder_names = ["hifi_gan"] 

        
    titles = ['sample_path', 'masked_WER', 'masked_num_wrong_words', 'total_masked_num_words', 'true_trans', 'masked_trans', 'masked_plcmos',
                'masked_OVRL_raw', 'masked_SIG_raw', 'masked_BAK_raw', 'masked_OVRL', 'masked_SIG', 'masked_BAK', 'masked_P808_MOS']
    titles = titles + [met + '_' + voc for met in ['target_OVRL_raw', 'target_SIG_raw', 'target_BAK_raw', 'target_OVRL', 'target_SIG', 'target_BAK', 'target_P808_MOS'] for voc in vocoder_names] + \
                [met + '_' + voc for met in ['OVRL_raw', 'SIG_raw', 'BAK_raw', 'OVRL', 'SIG', 'BAK', 'P808_MOS'] for voc in vocoder_names] + \
                [met + '_' + voc for met in ['plcmos_target_init', 'LSD_init', 'STOI_init', 'PESQ_init'] for voc in vocoder_names] + \
                [met + '_' + voc for met in ['WER', 'trans', 'num_wrong_words', 'total_num_words', 'plcmos_pred', 'LSD', 'STOI', 'PESQ'] for voc in vocoder_names]
    if not mel_text_bool:
        titles.append('mel_text_false_wer')
    compute_metrics = Metrics()
    dict_row_results = {}
    sample_folders = []
    for base_path in pathes2data:
        sample_folders.extend([f for f in glob.glob(os.path.join(base_path, '**', 'sample*'), recursive=True) if os.path.isdir(f)])
    num_dirs = len(sample_folders)
    print(f"There is a total of {num_dirs} directories")
    for i, path2sample_dir in enumerate(sample_folders):
        # if "sample_312" not in path2sample_dir:
        #     continue
        # Open the file once and keep it open
        with open(save_csv_path, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=titles, delimiter="|")
            if i == 0:
                writer.writeheader()  # Write the header row
            dict_row_results["sample_path"] = Path(path2sample_dir).relative_to('/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/')
            with open(f'{path2sample_dir}/asr_text.txt', "r") as file:
                lines = file.readlines()
                # Extract the sentences
                target_text = lines[0].split(":")[1].strip()
                if not mel_text_bool:
                    asrplm_text = lines[1].split(":")[1].strip()

                
            masked_audio, sr = torchaudio.load(f'{path2sample_dir}/masked_audio_time.wav') #masked_audio_time, time_masking_audio
            masked_audio = masked_audio[0].to(device)
            dict_row_results['masked_WER'], dict_row_results["total_masked_num_words"], dict_row_results["masked_num_wrong_words"], \
                dict_row_results["true_trans"], dict_row_results["masked_trans"] = compute_metrics.calc_wer(target_text, masked_audio)
            if not mel_text_bool:
                dict_row_results["mel_text_false_wer"] = compute_metrics.calc_wer_2_texts(target_text, asrplm_text)
            dict_row_results['masked_plcmos'] = round(compute_metrics.calc_plcmos(masked_audio), 4)
            dict_row_results.update({'masked_' + k: round(v, 4) for k, v in compute_metrics.calc_dnsmos(masked_audio).items()})
            for voc in vocoder_names:
                
                pred, sr = torchaudio.load(f'{path2sample_dir}/generated_audio_{voc}.wav')
                tar_wav, sr = torchaudio.load(f'{path2sample_dir}/gt_audio_{voc}.wav')

                pred = pred[0].to(device)
                tar_wav = tar_wav[0].to(device)
                
                metrics = compute_metrics.compute_metrics(pred, tar_wav, target_text)
                for key, value in metrics.items():
                    dict_row_results[key + '_' + voc] = round(value, 4) if is_number(value) else value
                # print(metrics)
                metrics = compute_metrics.compute_init_metrics(masked_audio, tar_wav)
                for key, value in metrics.items():
                    dict_row_results[key + '_' + voc] = round(value, 4) if is_number(value) else value
                
                metrics = compute_metrics.calc_dnsmos(pred)
                for key, value in metrics.items():
                    dict_row_results[key + '_' + voc] = round(value, 4) if is_number(value) else value
                    
                metrics = compute_metrics.calc_dnsmos(tar_wav)
                for key, value in metrics.items():
                    dict_row_results['target_' + key + '_' + voc] = round(value, 4) if is_number(value) else value

            writer.writerow(dict_row_results)
            print('\n -------------------')
            # if idx == 3:
            #     break
        
        # if combine_dataframes:
        #     output_csv = f"{path2data}/metric_results_and_samples_info.csv"

        #     # Convert new samples to a DataFrame
        #     df_new = pd.read_csv(save_csv_path, delimiter="|")

        #     input_csv = f"{path2data}/samples_info.csv" #samples_info
        #     df_existing = pd.read_csv(input_csv, delimiter="|")
        #     # Concatenate the two DataFrames along the rows (axis=0)
        #     df_combined = pd.concat([df_existing, df_new], ignore_index=False, axis=1)

        #     # Save the combined DataFrame to a new CSV
        #     df_combined.to_csv(output_csv, index=False)
    
if __name__ == '__main__':
    mel_text_bool = False
    name_csv = 'dit_mel-text=False_g2p-no-nn_lm-weight=0p3_ctc-weight=0p1_WER_skip=50_test'
    pathes2data = ['/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/DiT_Anechoic_LibSp_conditional-masked-melspec_w-masked-pix=1/dit-net_dim768_depth18_heads12_dim-head64_dropout0.1_ff_mult2_T400_betaT0.02/repeat_all_freq-length=20_skip=50_cp=112000_mel_text=False_phoneme-without-space_g2p-no-nn_lm-weight=0p3_ctc-weight=0p1']
    main(pathes2data, name_csv=name_csv, mel_text_bool=mel_text_bool)