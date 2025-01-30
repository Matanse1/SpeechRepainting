
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
        return wer, num_words_ref, wrong_words, reference_clean, hypo_clean
    
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


def main():
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
    pathes2data    =        ['/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/Unet_Anechoic_LibSp_unconditional/unet_dim64_dim_mults1_2_4_T400_betaT0.02/medium-gap_cp=1256000_mel_text=True_withoutLM/w1=2_w2=0.8_asr_start=320_mask=True',
                   '/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/Unet_Anechoic_LibSp_unconditional/unet_dim64_dim_mults1_2_4_T400_betaT0.02/small-gap_cp=1256000_mel_text=True_withoutLM/w1=2_w2=0.8_asr_start=320_mask=True']
    for path2data in pathes2data:
    # path2data = '/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/Unet_Anechoic_LibSp_wavlm-conditional_w-masked-pix=0.8/unet_dim64_dim_mults1_2_4_T400_betaT0.02/as-train-gap_cp=532000_mel_text=True_withoutLM/w1=2_w2=0.8_asr_start=320_mask=True'
        num_dirs = count_directories(path2data)
        print(f"There is a total of {num_dirs} directories")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vocoder_names = ["bigvgan", "hifi_gan"]
        titles = ['masked_WER', 'masked_num_wrong_words', 'total_masked_num_words', 'true_trans', 'masked_trans', 'masked_plcmos']
        titles = titles + [met + '_' + voc for met in ['plcmos_target_init', 'LSD_init', 'STOI_init', 'PESQ_init'] for voc in vocoder_names] + \
                    [met + '_' + voc for met in ['WER', 'trans', 'num_wrong_words', 'total_num_words', 'plcmos_pred', 'LSD', 'STOI', 'PESQ'] for voc in vocoder_names]
                    
        compute_metrics = Metrics()
        dict_row_results = {}
        save_csv_path = f"{path2data}/metric_results.csv"
        # Open the file once and keep it open
        with open(save_csv_path, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=titles, delimiter="|")
            writer.writeheader()  # Write the header row
            for sample in tqdm(range(num_dirs)):
                with open(f'{path2data}/sample_{sample}/asr_text.txt', "r") as file:
                    lines = file.readlines()
                    # Extract the sentences
                    target_text = lines[0].split(":")[1].strip()
                    estimated_text = lines[1].split(":")[1].strip()
                masked_audio, sr = torchaudio.load(f'{path2data}/sample_{sample}/masked_audio_time.wav') #masked_audio_time, time_masking_audio
                masked_audio = masked_audio[0].to(device)
                dict_row_results['masked_WER'], dict_row_results["masked_num_wrong_words"], dict_row_results["total_masked_num_words"], \
                    dict_row_results["true_trans"], dict_row_results["masked_trans"] = compute_metrics.calc_wer(target_text, masked_audio)
                dict_row_results['masked_plcmos'] = round(compute_metrics.calc_plcmos(masked_audio), 3)
                for voc in vocoder_names:
                    
                    pred, sr = torchaudio.load(f'{path2data}/sample_{sample}/generated_audio_{voc}.wav')
                    tar_wav, sr = torchaudio.load(f'{path2data}/sample_{sample}/gt_audio_{voc}.wav')

                    pred = pred[0].to(device)
                    tar_wav = tar_wav[0].to(device)
                    
                    metrics = compute_metrics.compute_metrics(pred, tar_wav, target_text)
                    for key, value in metrics.items():
                        dict_row_results[key + '_' + voc] = value
                    # print(metrics)
                    metrics = compute_metrics.compute_init_metrics(masked_audio, tar_wav)
                    for key, value in metrics.items():
                        dict_row_results[key + '_' + voc] = value

                writer.writerow(dict_row_results)
                print('-------------------')
                # if sample == 2:
                #     break

        output_csv = f"{path2data}/metric_results_and_samples_info.csv"

        # Convert new samples to a DataFrame
        df_new = pd.read_csv(save_csv_path, delimiter="|")

        input_csv = f"{path2data}/samples_info.csv" #samples_info
        df_existing = pd.read_csv(input_csv, delimiter="|")
        # Concatenate the two DataFrames along the rows (axis=0)
        df_combined = pd.concat([df_existing, df_new], ignore_index=False, axis=1)

        # Save the combined DataFrame to a new CSV
        df_combined.to_csv(output_csv, index=False)
    
if __name__ == '__main__':
    main()