
# the mos is from https://github.com/alessandroragano/scoreq/tree/bc1d19894092129f5dff774a0c9d942ac626d2a1,
# paper titled "SCOREQ:Speech Quality Assessment with Contrastive Regression"
import scoreq
# from prettytable import PrettyTable
import glob
import os
import csv
from pathlib import Path
# import torchaudio
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
def main(pathes2data, csv_name='all'):


    print("Load models")
    if mos_bool:
        # Predict quality of natural speech in NR mode
        nr_scoreq_n = scoreq.Scoreq(data_domain='natural', mode='nr')
    if distance_mos_bool:
        # Predict quality of natural speech in REF mode
        ref_scoreq_n = scoreq.Scoreq(data_domain='natural', mode='ref')



    print("Finished load models")
    
    # Paths for different data sources
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
    if mos_bool:
        titles = titles + ["masked_mos"]
        titles = titles + [met + '_' + voc for met in ['mos', 'mos_init'] for voc in vocoder_names]
    if distance_mos_bool:
        titles = titles + [met + '_' + voc for met in ['distance_mos', 'masked_distance_mos'] for voc in vocoder_names]
    

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
    interval_save = 10000  # Define the interval to save results
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
                    sample = Path(sample_path).name.split('_')[-1]

                    
                    masked_path = f'{sample_path}/masked_audio_time.wav'
                    if mos_bool:
                        pred_mos = nr_scoreq_n.predict(test_path=masked_path, ref_path=None)
                        pred_mos = round(pred_mos, 4)
                        dict_row_results["masked_mos"] = pred_mos 
                                    
                        for voc in vocoder_names:
                            speech_target_path = f'{sample_path}/gt_audio_{voc}.wav'
                            speech_est_path = f'{sample_path}/generated_audio_{voc}.wav'
                            pred_mos = nr_scoreq_n.predict(test_path=speech_target_path, ref_path=None)
                            pred_mos = round(pred_mos, 4)
                            dict_row_results["mos_init_" + voc] = pred_mos
                            pred_mos = nr_scoreq_n.predict(test_path=speech_est_path, ref_path=None)
                            pred_mos = round(pred_mos, 4)
                            dict_row_results["mos_" + voc] = pred_mos
                    if distance_mos_bool:
                        for voc in vocoder_names:
                            speech_target_path = f'{sample_path}/gt_audio_{voc}.wav'
                            speech_est_path = f'{sample_path}/generated_audio_{voc}.wav'
                            pred_distance_mos = ref_scoreq_n.predict(test_path=speech_est_path, ref_path=speech_target_path)
                            pred_distance_mos = round(pred_distance_mos, 4)
                            dict_row_results["distance_mos_" + voc] = pred_distance_mos
                            pred_distance_mos = ref_scoreq_n.predict(test_path=masked_path, ref_path=speech_target_path)
                            pred_distance_mos = round(pred_distance_mos, 4)
                            dict_row_results["masked_distance_mos_" + voc] = pred_distance_mos

                    # Write the results to the CSV file
                    writer.writerow(dict_row_results)
                except Exception as e:
                    print(f"Error processing {sample_path}: {e}")
                    continue

        print(f"Processed interval {i // interval_save + 1}")
    print("Finished processing all sample folders")

if __name__ == '__main__':
    pathes2data = ['/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/DiT_Anechoic_LibSp_conditional-masked-melspec_w-masked-pix=1/dit-net_dim768_depth18_heads12_dim-head64_dropout0.1_ff_mult2_T400_betaT0.02/repeat_all_freq-length=10_skip=25_cp=112000_mel_text=False_phoneme-without-space_g2p-no-nn_lm-weight=0p5_ctc-weight=0p1_bs=80',
                   '/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/DiT_Anechoic_LibSp_conditional-masked-melspec_w-masked-pix=1/dit-net_dim768_depth18_heads12_dim-head64_dropout0.1_ff_mult2_T400_betaT0.02/repeat_all_freq-length=20_skip=50_cp=112000_mel_text=False_phoneme-without-space_g2p-no-nn_lm-weight=0p5_ctc-weight=0p1_bs=80',
                   '/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/DiT_Anechoic_LibSp_conditional-masked-melspec_w-masked-pix=1/dit-net_dim768_depth18_heads12_dim-head64_dropout0.1_ff_mult2_T400_betaT0.02/repeat_all_freq-length=30_skip=75_cp=112000_mel_text=False_phoneme-without-space_g2p-no-nn_lm-weight=0p5_ctc-weight=0p1_bs=80']
    csv_name = 'dit_mel-text=False_g2p-no-nn_length=10+20+30_skip=25+50+75_lm=0p5_ctc=0p1_bs=80_distance-mos'
    distance_mos_bool = True
    mos_bool = False
    main(pathes2data, csv_name=csv_name)