

# the mos is from https://github.com/alessandroragano/scoreq/tree/bc1d19894092129f5dff774a0c9d942ac626d2a1,
# paper titled "SCOREQ:Speech Quality Assessment with Contrastive Regression"

import torchaudio
print(torchaudio.list_audio_backends())
reference_audio_path = '/home/dsi/moradim/SpeechRepainting/my_utils/sample_378_long/gt_audio_hifi_gan.wav'  # Replace with the path to your reference audio file
degraded_audio_path = '/home/dsi/moradim/SpeechRepainting/my_utils/sample_378_long/generated_audio_hifi_gan.wav' 
degraded_audio_path_worse1 = '/home/dsi/moradim/SpeechRepainting/my_utils/sample_378_long/generated_audio_hifi_gan_worse1.wav'
masked_audio_path = '/home/dsi/moradim/SpeechRepainting/my_utils/sample_378_long/masked_audio_time.wav'

# reference_audio_path = '/home/dsi/moradim/SpeechRepainting/my_utils/sample_378/gt_audio_hifi_gan.wav'  # Replace with the path to your reference audio file
# degraded_audio_path = '/home/dsi/moradim/SpeechRepainting/my_utils/sample_378/generated_audio_hifi_gan_better.wav' #'/home/dsi/moradim/SpeechRepainting/my_utils/sample_378/generated_audio_hifi_gan.wav' 
# degraded_audio_path_worse1 = '/home/dsi/moradim/SpeechRepainting/my_utils/sample_378/generated_audio_hifi_gan_worse1.wav'
# masked_audio_path = '/home/dsi/moradim/SpeechRepainting/my_utils/sample_378/masked_audio_time.wav'

a = torchaudio.load(degraded_audio_path)
import scoreq
from prettytable import PrettyTable
dict_path = {'Worse':degraded_audio_path_worse1, 'Better':degraded_audio_path, 'Masked':masked_audio_path, 'Reference':reference_audio_path}
dict_results = {'Worse':{}, 'Better':{}, 'Masked':{}, 'Reference':{}}

print("Load models")
# Predict quality of natural speech in NR mode
nr_scoreq_n = scoreq.Scoreq(data_domain='natural', mode='nr')
# Predict quality of natural speech in REF mode
ref_scoreq_n = scoreq.Scoreq(data_domain='natural', mode='ref')
# Predict quality of synthetic speech in NR mode
nr_scoreq_s = scoreq.Scoreq(data_domain='synthetic', mode='nr')
# Predict quality of synthetic speech in REF mode
ref_scoreq_s = scoreq.Scoreq(data_domain='synthetic', mode='ref')

# Create a table to display the results
table = PrettyTable()
table.field_names = ["Condition", "Data Domain", "Mode", "Predicted MOS", "Predicted Distance"]
print("-------------------------")
for key in dict_path.keys():
    print(f"Predicting on {key}")
    reference_audio_path = dict_path['Reference']
    degraded_audio_path = dict_path[key]
    # Predict quality of natural speech in NR mode
    pred_mos_n = nr_scoreq_n.predict(test_path=degraded_audio_path, ref_path=None)
    print(f"The pred mos = {pred_mos_n}")
    table.add_row([key, "Natural", "NR", pred_mos_n, "N/A"])
    # Predict quality of natural speech in REF mode
    pred_distance_n = ref_scoreq_n.predict(test_path=degraded_audio_path, ref_path=reference_audio_path)
    print(f"The pred_distance = {pred_distance_n}")
    table.add_row([key, "Natural", "REF", "N/A", pred_distance_n])
    # Predict quality of synthetic speech in NR mode
    pred_mos_s = nr_scoreq_s.predict(test_path=degraded_audio_path, ref_path=None)
    print(f"The pred mos = {pred_mos_n}")
    table.add_row([key, "Synthetic", "NR", pred_mos_s, "N/A"])
    # Predict quality of synthetic speech in REF mode
    pred_distance_s = ref_scoreq_s.predict(test_path=degraded_audio_path, ref_path=reference_audio_path)
    print(f"The pred_distance = {pred_distance_s}")
    table.add_row([key, "Synthetic", "REF", "N/A", pred_distance_s])




# Print the table
print(table)