
# the mos is from https://github.com/Takaaki-Saeki/DiscreteSpeechMetrics/tree/master,
# paper titled 

# from prettytable import PrettyTable
import glob
import os
import csv
from pathlib import Path
# import torchaudio
from tqdm import tqdm
from discrete_speech_metrics import MCD
from discrete_speech_metrics import SpeechTokenDistance
from discrete_speech_metrics import SpeechBLEU
from discrete_speech_metrics import SpeechBERTScore
from discrete_speech_metrics import LogF0RMSE
from discrete_speech_metrics import UTMOS
import soundfile as sf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def main(pathes2data, csv_name='all', interval_save=4500):


    print("Load models")
    if logfmse_bool:
        metrics_f0_mse = LogF0RMSE(sr=16000)
    if mcd_bool:
        metrics_mcd = MCD(sr=16000)
    if sptd:
        metrics_sptd = SpeechTokenDistance(
            sr=16000,
            model_type="hubert-base",
            vocab=200,
            layer=6,
            distance_type="jaro-winkler",
            remove_repetition=False,
            use_gpu=True)
    if spbl:
        n_ngram = 5
        metrics_spbl = SpeechBLEU(
            sr=16000,
            model_type="hubert-base",
            vocab=200,
            layer=11,
            n_ngram=n_ngram,
            remove_repetition=True,
            use_gpu=True)


    if spbc:
        metrics_spbc = SpeechBERTScore(
            sr=16000,
            model_type="wavlm-large",
            layer=14,
            use_gpu=True)
    
    if utmos_bool:
        metrics_utmos = UTMOS(sr=16000)

    print("Finished load models")
    
    # Paths for different data sources
    # pathes2data = ['/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/DiT_Anechoic_LibSp_conditional-masked-melspec_w-masked-pix=1/dit-net_dim768_depth18_heads12_dim-head64_dropout0.1_ff_mult2_T400_betaT0.02']
    # pathes2data = ['/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/DiT_Anechoic_LibSp_conditional-masked-melspec_w-masked-pix=1/dit-net_dim768_depth18_heads12_dim-head64_dropout0.1_ff_mult2_T400_betaT0.02']

    # List to store all found sample folder paths
    sample_folders = []

    # Iterate over each path in the list
    for base_path in pathes2data:
        # Use glob to recursively search for folders starting with 'sample'
        sample_folders.extend([f for f in glob.glob(os.path.join(base_path, '**', 'sample*'), recursive=True) if os.path.isdir(f)])

    print(f"Found {len(sample_folders)} sample folders")

    # Vocoder names
    # vocoder_names = ["bigvgan", "hifi_gan"]
    vocoder_names = ["hifi_gan"]

    # Define titles for the CSV file 
    titles = ['sample_path']
    if utmos_bool:
        titles = titles + ['masked_utmos']
        titles = titles + [met + '_' + voc for met in ['utmos', 'utmos_init'] for voc in vocoder_names]
    if mcd_bool:
        titles = titles + [met + '_' + voc for met in ['mcd', 'masked_mcd'] for voc in vocoder_names]
    if logfmse_bool:
        titles = titles + [met + '_' + voc for met in ['masked_logf0rmse', 'logf0rmse'] for voc in vocoder_names]
    if spbl:
        titles = titles + [met + '_' + voc for met in ['speech_bleu', 'masked_speech_bleu'] for voc in vocoder_names]
    if spbc:
        titles = titles + [met + '_' + voc for met in ['speech_bert', 'masked_speech_bert'] for voc in vocoder_names]
    if sptd:
        titles = titles + [met + '_' + voc for met in ['speech_token_distance', 'masked_speech_token_distance'] for voc in
                           vocoder_names]

    

    # CSV save path
    save_csv_path = f"/home/dsi/moradim/SpeechRepainting/metric_results_{csv_name}.csv"

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

    # Process the sample folders in intervals
    for i in tqdm(range(0, len(sample_folders), interval_save)):
        sample_folders_interval = sample_folders[i:i + interval_save]
        
        with open(save_csv_path, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=titles, delimiter="|")
            if i == 0:
                writer.writeheader()

            # Process each sample folder within the interval
            for sample_path in tqdm(sample_folders_interval):
                # if ("repeat_all_freq-length" not in sample_path) or ("cp=112000" not in sample_path) or ("noise" in sample_path):
                #     print(f"Skipping {sample_path}")
                #     continue
                try:
                    dict_row_results = {}
                    dict_row_results["sample_path"] = Path(sample_path).relative_to('/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/')  
                    # dict_row_results["sample_path"] = Path(sample_path).relative_to('/home/dsi/moradim/SpeechRepainting')     
                            
                    masked_path = f'{sample_path}/masked_audio_time.wav'
                    masked_wav, sr = sf.read(masked_path)
                    if utmos_bool:
                        utmos_m = metrics_utmos.score(masked_wav)
                        utmos_m = round(utmos_m, 4)
                        dict_row_results["masked_utmos"] = utmos_m
                                    
                    for voc in vocoder_names:
                        speech_target_path = f'{sample_path}/gt_audio_{voc}.wav'
                        ref_wav, sr = sf.read(speech_target_path)
                        gen_wav_path = f'{sample_path}/generated_audio_{voc}.wav'
                        gen_wav, sr = sf.read(gen_wav_path)
                        
                        if utmos_bool:
                            print("utmos")
                            utmos = metrics_utmos.score(gen_wav)
                            utmos_init = metrics_utmos.score(ref_wav)
                            dict_row_results[f"utmos_{voc}"] = round(utmos, 4)
                            dict_row_results[f"utmos_init_{voc}"] = round(utmos_init, 4)
                            print("utmos done")
                            
                        if spbc:
                            print("speech_bert")
                            precision, _, _ = metrics_spbc.score(ref_wav, gen_wav)
                            precision_masked, _, _ = metrics_spbc.score(ref_wav, masked_wav)
                            dict_row_results[f"speech_bert_{voc}"] = round(precision, 4)
                            dict_row_results[f"masked_speech_bert_{voc}"] = round(precision_masked, 4)
                            print("speech_bert done")
                        
                        if spbl:
                            # print("speech_bleu")
                            bleu_masked = metrics_spbl.score(ref_wav, masked_wav)
                            bleu = metrics_spbl.score(ref_wav, gen_wav)
                            dict_row_results[f"speech_bleu_{voc}"] = round(bleu, 4)
                            dict_row_results[f"masked_speech_bleu_{voc}"] = round(bleu_masked, 4)
                            # print("speech_bleu done")
                        
                        if sptd:
                            print("speech_token_distance")
                            distance = metrics_sptd.score(ref_wav, gen_wav)
                            distance_m = metrics_sptd.score(ref_wav, masked_wav)
                            dict_row_results[f"speech_token_distance_{voc}"] = round(distance, 4)
                            dict_row_results[f"masked_speech_token_distance_{voc}"] = round(distance_m, 4)
                            print("speech_token_distance done")
                        if mcd_bool:
                            # print("mcd")
                            mcd = metrics_mcd.score(ref_wav, gen_wav)
                            mcd_m = metrics_mcd.score(ref_wav, masked_wav)
                            dict_row_results[f"mcd_{voc}"] = round(mcd, 4)
                            dict_row_results[f"masked_mcd_{voc}"] = round(mcd_m, 4)
                            # print("mcd done")
                            
                        if logfmse_bool:
                            # print("logf0rmse")
                            logf0rmse = metrics_f0_mse.score(ref_wav, gen_wav)
                            logf0rmse_m = metrics_f0_mse.score(ref_wav, masked_wav)
                            dict_row_results[f"logf0rmse_{voc}"] = round(logf0rmse, 4)
                            dict_row_results[f"masked_logf0rmse_{voc}"] = round(logf0rmse_m, 4)
                            # print("logf0rmse done")

                    # Write the results to the CSV file
                    if len(dict_row_results) > 0:
                        writer.writerow(dict_row_results)
                except Exception as e:
                    print(f"Error processing {sample_path}: {e}")
                    continue

        print(f"Processed interval {i // interval_save + 1}")
    print("Finished processing all sample folders")

if __name__ == '__main__':
    utmos_bool = False
    mcd_bool = True
    logfmse_bool = True
    spbl = True
    spbc = False
    sptd = False
    
    pathes2data = ['/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/DiT_Anechoic_LibSp_conditional-masked-melspec_w-masked-pix=1/dit-net_dim768_depth18_heads12_dim-head64_dropout0.1_ff_mult2_T400_betaT0.02/repeat_all_freq-length=10_skip=150_cp=112000_mel_text=False_phoneme-without-space_g2p-no-nn',
                   '/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/DiT_Anechoic_LibSp_conditional-masked-melspec_w-masked-pix=1/dit-net_dim768_depth18_heads12_dim-head64_dropout0.1_ff_mult2_T400_betaT0.02/repeat_all_freq-length=20_skip=150_cp=112000_mel_text=False_phoneme-without-space_g2p-no-nn',
                   '/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/DiT_Anechoic_LibSp_conditional-masked-melspec_w-masked-pix=1/dit-net_dim768_depth18_heads12_dim-head64_dropout0.1_ff_mult2_T400_betaT0.02/repeat_all_freq-length=30_skip=150_cp=112000_mel_text=False_phoneme-without-space_g2p-no-nn',
                   '/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/DiT_Anechoic_LibSp_conditional-masked-melspec_w-masked-pix=1/dit-net_dim768_depth18_heads12_dim-head64_dropout0.1_ff_mult2_T400_betaT0.02/repeat_all_freq-length=40_skip=150_cp=112000_mel_text=False_phoneme-without-space_g2p-no-nn',
                   '/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/DiT_Anechoic_LibSp_conditional-masked-melspec_w-masked-pix=1/dit-net_dim768_depth18_heads12_dim-head64_dropout0.1_ff_mult2_T400_betaT0.02/repeat_all_freq-length=50_skip=150_cp=112000_mel_text=False_phoneme-without-space_g2p-no-nn',]
    interval_save = 10000
    csv_name = 'dit_mel-text=False_g2p-no-nn_lm-weight=0p3_ctc-weight=0p1_speech_bleu_n-gram=5'
    main(pathes2data, csv_name=csv_name, interval_save=interval_save)
    




