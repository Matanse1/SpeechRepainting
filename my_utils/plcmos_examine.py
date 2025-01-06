import soundfile as sf
import sys
sys.path.append('/home/dsi/moradim/SpeechRepainting')
from PLCMOS.plc_mos import PLCMOSEstimator

plcmos = PLCMOSEstimator()
    
target_path = '/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/LibSp_wavlm-base-plus-rep_w_masked_pix=0.8_two_branch=True_specific_hidden_states/wnet_h512_d12_T400_betaT0.02/as-train-gap_asr_guidance_80cp_filled-with-randn/w1=2_w2=1.5_asr_start=270_mask=True/sample_0/gt_audio_hifi_gan.wav'
masked_path = '/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/LibSp_wavlm-base-plus-rep_w_masked_pix=0.8_two_branch=True_specific_hidden_states/wnet_h512_d12_T400_betaT0.02/as-train-gap_asr_guidance_80cp_filled-with-randn/w1=2_w2=1.5_asr_start=270_mask=True/sample_0/masked_audio_time.wav'
est_path = '/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/LibSp_wavlm-base-plus-rep_w_masked_pix=0.8_two_branch=True_specific_hidden_states/wnet_h512_d12_T400_betaT0.02/as-train-gap_asr_guidance_80cp_filled-with-randn/w1=2_w2=1.5_asr_start=270_mask=True/sample_0/generated_audio_hifi_gan.wav'
data, sr = sf.read(target_path)
mos = plcmos.run(data, sr)
print(f"For target: {mos}")

data, sr = sf.read(masked_path)
mos = plcmos.run(data, sr)
print(f"For masked: {mos}")

data, sr = sf.read(est_path)
mos = plcmos.run(data, sr)
print(f"For est: {mos}")