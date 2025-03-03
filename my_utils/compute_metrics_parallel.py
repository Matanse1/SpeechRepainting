
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
import multiprocessing
import jiwer
from whisper.normalizers import EnglishTextNormalizer
import glob
import os
import csv
import pandas as pd
from tqdm import tqdm
import cProfile
from pathlib import Path
import torch.multiprocessing as mp
import concurrent.futures
import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Manager

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
    def __init__(self, sampling_rate=16000, whisper_model_name= 'medium.en', path2whisper_model='/dsi/gannot-lab1/users/mordehay/whisper_models', device='cuda:0'):
        self.sampling_rate = sampling_rate
        self.stoi = STOI(sampling_rate)
        self.pesq = PESQ(sampling_rate, 'wb') # wide band signal (speech)
        self.plcmos = PLCMOSEstimator()
        self.whisper_model = whisper.load_model(whisper_model_name, download_root=path2whisper_model, device=device) # choices=['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large'], default='medium.en', help='Pretrained ASR model. Those with .en are english-only')
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


def test_func(x):
    print(f"Process {os.getpid()} processing {x}")

def init_worker(gpu_dict_var = {0: 4, 1:1}):
    global compute_metrics  # Import inside worker
    global gpu_dict
    gpu_dict = gpu_dict_var
    # global gpu_num
    # Get process index (derive from process name)
    process_name = multiprocessing.current_process().name  # Example: "Process-2"
    process_id = int(process_name.split('-')[-1]) - 1  # Convert to zero-based index
    print(f"Process {process_id} initializing")
    gpu_num = process_id % len(gpu_dict)
    
    gpu_num = gpu_dict[gpu_num]
    compute_metrics = Metrics(device=f"cuda:{gpu_num}")  # Initialize in worker process
                      
def process_wav(sample_path):
    # if not os.path.isdir(sample_path):
    #     return None
    process_name = multiprocessing.current_process().name  # Example: "Process-2"
    process_id = int(process_name.split('-')[-1]) - 1
    gpu_num = process_id % len(gpu_dict)
    # gpu_dict = {0: 4, 1:1}
    gpu_num = gpu_dict[gpu_num]
    # print("Inside process_wav")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = f"cuda:{gpu_num}"
    vocoder_names = ["bigvgan", "hifi_gan"]
    dict_row_results = {}
    dict_row_results["sample_path"] = Path(sample_path).relative_to('/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/')
    sample = Path(sample_path).name.split('_')[-1]
    with open(f'{sample_path}/asr_text.txt', "r") as file:
        lines = file.readlines()
        # Extract the sentences
        target_text = lines[0].split(":")[1].strip()
        # estimated_text = lines[1].split(":")[1].strip()
    masked_audio, sr = torchaudio.load(f'{sample_path}/masked_audio_time.wav') #masked_audio_time, time_masking_audio
    masked_audio = masked_audio[0].to(device)
    dict_row_results['masked_WER'], dict_row_results["total_masked_num_words"], dict_row_results["masked_num_wrong_words"], \
        dict_row_results["true_trans"], dict_row_results["masked_trans"] = compute_metrics.calc_wer(target_text, masked_audio)
    dict_row_results['masked_plcmos'] = round(compute_metrics.calc_plcmos(masked_audio), 4)
    dict_row_results.update({'masked_' + k: round(v, 4) for k, v in compute_metrics.calc_dnsmos(masked_audio).items()})
    for voc in vocoder_names:
        
        pred, sr = torchaudio.load(f'{sample_path}/generated_audio_{voc}.wav')
        tar_wav, sr = torchaudio.load(f'{sample_path}/gt_audio_{voc}.wav')

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

        
    return dict_row_results
    # # Locking the CSV writing to ensure thread safety
    # with csv_writer_lock:
    #     csv_writer.writerow(dict_row_results)

def main(interval_save=5, max_workers=4, gpu_dict_var={0: 4, 1:1}, specifc_folder=True):
    mp.set_start_method("spawn", force=True)
    if specifc_folder:
        pathes2data = ['/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/DiT_Anechoic_LibSp_conditional-masked-melspec_w-masked-pix=1/dit-net_dim768_depth18_heads12_dim-head64_dropout0.1_ff_mult2_T400_betaT0.02/repeat_all_freq-length=25_skip=150_cp=112000_mel_text=True_ASR/w1=-1_w2=0.5_asr_start=270_mask=True',
                    '/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/DiT_Anechoic_LibSp_conditional-masked-melspec_w-masked-pix=1/dit-net_dim768_depth18_heads12_dim-head64_dropout0.1_ff_mult2_T400_betaT0.02/repeat_all_freq-length=25_skip=150_cp=112000_mel_text=True_ASR/w1=-1_w2=0.5_asr_start=320_mask=True']
        # List to store all found sample folder paths
        sample_folders = []

        # Iterate over each path in the list
        for base_path in pathes2data:
            # Use glob to recursively search for folders named 'sample'
            # sample_folders.extend(glob.glob(os.path.join(base_path, '**', 'sample'), recursive=True))
            sample_folders.extend([f for f in glob.glob(os.path.join(base_path, 'sample*')) if os.path.isdir(f)])
    else:
        pathes2data = ['/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/DiT_Anechoic_LibSp_conditional-masked-melspec_w-masked-pix=1/dit-net_dim768_depth18_heads12_dim-head64_dropout0.1_ff_mult2_T400_betaT0.02']
        # List to store all found sample folder paths
        sample_folders = []

        # Iterate over each path in the list
        for base_path in pathes2data:
            # Use glob to recursively search for folders named 'sample'
            # sample_folders.extend(glob.glob(os.path.join(base_path, '**', 'sample'), recursive=True))
            sample_folders.extend([f for f in glob.glob(os.path.join(base_path, '**', 'sample*'), recursive=True) if os.path.isdir(f)])
    print(f"Found {len(sample_folders)} sample folders")
    vocoder_names = ["bigvgan", "hifi_gan"] 

        
    titles = ['sample_path', 'masked_WER', 'masked_num_wrong_words', 'total_masked_num_words', 'true_trans', 'masked_trans', 'masked_plcmos',
                'masked_OVRL_raw', 'masked_SIG_raw', 'masked_BAK_raw', 'masked_OVRL', 'masked_SIG', 'masked_BAK', 'masked_P808_MOS']
    titles = titles + [met + '_' + voc for met in ['target_OVRL_raw', 'target_SIG_raw', 'target_BAK_raw', 'target_OVRL', 'target_SIG', 'target_BAK', 'target_P808_MOS'] for voc in vocoder_names] + \
                [met + '_' + voc for met in ['OVRL_raw', 'SIG_raw', 'BAK_raw', 'OVRL', 'SIG', 'BAK', 'P808_MOS'] for voc in vocoder_names] + \
                [met + '_' + voc for met in ['plcmos_target_init', 'LSD_init', 'STOI_init', 'PESQ_init'] for voc in vocoder_names] + \
                [met + '_' + voc for met in ['WER', 'trans', 'num_wrong_words', 'total_num_words', 'plcmos_pred', 'LSD', 'STOI', 'PESQ'] for voc in vocoder_names]
                
    # compute_metrics = Metrics()
    save_csv_path = f"/home/dsi/moradim/SpeechRepainting/metric_results.csv"
    user_input = input(f"Do you want to remove the existing file at {save_csv_path}? (yes/no): ")

    if user_input.lower() == 'yes':
        # 🔹 Remove the CSV file
        csv_path = Path(save_csv_path)
        if csv_path.exists():
            csv_path.unlink()  # Remove the file
            print(f"File {save_csv_path} has been removed.")
        else:
            print(f"File {save_csv_path} does not exist.")
    else:
        print(f"Proceeding without removing the file.")
    # Open the file once and keep it open

    # sample_folders = sample_folders[:4]
    for i in range(0, len(sample_folders), interval_save):
        sample_folders_interval = sample_folders[i:i+interval_save]
        with open(save_csv_path, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=titles, delimiter="|")
            if i == 0:
                writer.writeheader()
            
            # manager = Manager()
            # csv_writer_lock = manager.Lock()
            with ProcessPoolExecutor(initializer=init_worker, initargs=(gpu_dict_var,) ,max_workers=max_workers) as executor:
            # with ProcessPoolExecutor(initializer=init_worker, initargs=(compute_metrics,)) as executor:
            # with ProcessPoolExecutor() as executor:
            
                future_to_url = {executor.submit(process_wav, sample_folder): sample_folder for idx, sample_folder in enumerate(sample_folders_interval)}
                for future in tqdm(concurrent.futures.as_completed(future_to_url)):
                    clip = future_to_url[future]
                    try:
                        data = future.result()
                    except Exception as exc:
                        print('%r generated an exception: %s' % (clip, exc))
                    else:
                        if data is not None:
                            writer.writerow(data)
        
        print(f"Finished processing {i+interval_save} samples")
        
            # futures = []
            # for sample_folder in sample_folders:
            #     futures.append(executor.submit(process_wav, sample_folder))

            # # Wait for all futures to complete
            # for future in futures:
            #     future.result()
            
            
            #  with concurrent.futures.ThreadPoolExecutor() as executor:
            #     future_to_url = {executor.submit(compute_score, clip, desired_fs, is_personalized_eval): clip for clip in clips}
            #     for future in tqdm(concurrent.futures.as_completed(future_to_url)):
            #         clip = future_to_url[future]
            #         try:
            #             data = future.result()
            #         except Exception as exc:
            #             print('%r generated an exception: %s' % (clip, exc))
            #         else:
            #             rows.append(data) 
                # writer.writerow(future.result())
                  # Blocks until each process finishes
        # Using ProcessPoolExecutor for parallel processing
        # with ProcessPoolExecutor(max_workers=4) as executor:
        #     for sample_folder in sample_folders:
        #         # Submit jobs to the executor
        #         executor.submit(process_wav, sample_folder, csv_writer_lock, writer, compute_metrics)
                # executor.submit(test_func, sample_folder)
        # for sample_folder in tqdm(sample_folders):
        #     process_wav(sample_folder, csv_writer_lock, writer, compute_metrics)
    
if __name__ == '__main__':
    interval_save = 4500
    max_workers = 6
    gpu_dict_var = {0: 5, 1:6, 2:7}
    print(F"Using {len(gpu_dict_var)} GPUs")
    specifc_folder = False
    main(interval_save, max_workers, gpu_dict_var, specifc_folder)