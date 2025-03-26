
# the mos is from https://github.com/alessandroragano/scoreq/tree/bc1d19894092129f5dff774a0c9d942ac626d2a1,
# paper titled "SCOREQ:Speech Quality Assessment with Contrastive Regression"
import scoreq
from prettytable import PrettyTable
import glob
import os
import csv
from pathlib import Path
import torchaudio
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import torch.multiprocessing as mp

def init_worker(gpu_dict_var = {0: 4, 1:1}):
    global nr_scoreq_n  # Import inside worker
    global ref_scoreq_n  # Import inside worker
    global gpu_dict
    gpu_dict = gpu_dict_var
    # global gpu_num
    # Get process index (derive from process name)
    process_name = multiprocessing.current_process().name  # Example: "Process-2"
    process_id = int(process_name.split('-')[-1]) - 1  # Convert to zero-based index
    print(f"Process {process_id} initializing")
    gpu_num = process_id % len(gpu_dict)
    
    gpu_num = gpu_dict[gpu_num]
    # compute_metrics = Metrics(device=f"cuda:{gpu_num}")  # Initialize in worker process
    print("Load models")
    # Predict quality of natural speech in NR mode
    nr_scoreq_n = scoreq.Scoreq(data_domain='natural', mode='nr')
    # Predict quality of natural speech in REF mode
    ref_scoreq_n = scoreq.Scoreq(data_domain='natural', mode='ref')


def process_wav(sample_path):
    dict_row_results = {}
    dict_row_results["sample_path"] = Path(sample_path).relative_to('/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/')
    
    masked_path = f'{sample_path}/masked_audio_time.wav'
    pred_mos = nr_scoreq_n.predict(test_path=masked_path, ref_path=None)
    pred_mos = round(pred_mos, 4)
    dict_row_results["masked_mos"] = pred_mos 
    vocoder_names = ["bigvgan", "hifi_gan"]
    for voc in vocoder_names:
        speech_target_path = f'{sample_path}/gt_audio_{voc}.wav'
        speech_est_path = f'{sample_path}/generated_audio_{voc}.wav'
        pred_mos = nr_scoreq_n.predict(test_path=speech_target_path, ref_path=None)
        pred_mos = round(pred_mos, 4)
        dict_row_results["mos_init_" + voc] = pred_mos
        pred_mos = nr_scoreq_n.predict(test_path=speech_est_path, ref_path=None)
        pred_mos = round(pred_mos, 4)
        dict_row_results["mos_" + voc] = pred_mos
        
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
    return dict_row_results

def main(csv_name='all', interval_save=5, max_workers=4, gpu_dict_var = {0: 4, 1:1}):
    mp.set_start_method("spawn", force=True)
    # Paths for different data sources
    pathes2data = ['/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/DiT_Anechoic_LibSp_conditional-masked-melspec_w-masked-pix=1/dit-net_dim768_depth18_heads12_dim-head64_dropout0.1_ff_mult2_T400_betaT0.02']
    # pathes2data = ['/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/DiT_Anechoic_LibSp_conditional-masked-melspec_w-masked-pix=1/dit-net_dim768_depth18_heads12_dim-head64_dropout0.1_ff_mult2_T400_betaT0.02/repeat_all_freq-length=100_skip=150_cp=112000_mel_text=True_phoneme-without-space']
    # List to store all found sample folder paths
    sample_folders = []

    # Iterate over each path in the list
    for base_path in pathes2data:
        # Use glob to recursively search for folders starting with 'sample'
        sample_folders.extend([f for f in glob.glob(os.path.join(base_path, '**', 'sample*'), recursive=True) if os.path.isdir(f)])

    print(f"Found {len(sample_folders)} sample folders")

    # Vocoder names
    vocoder_names = ["bigvgan", "hifi_gan"]

    # Define titles for the CSV file
    titles = ['sample_path','masked_mos']
    titles = titles + [met + '_' + voc for met in ['mos', 'distance_mos', 'masked_distance_mos', 'mos_init'] for voc in vocoder_names]
    

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
        
    print("Finished processing all sample folders")

if __name__ == '__main__':
    csv_name = 'mos_dit'
    interval_save = 4500
    max_workers = 6
    gpu_dict_var = {0: 5, 1:6, 2:7}
    print(F"Using {len(gpu_dict_var)} GPUs")
    main(csv_name=csv_name, interval_save=interval_save, max_workers=max_workers, gpu_dict_var=gpu_dict_var)