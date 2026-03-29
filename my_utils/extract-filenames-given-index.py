import pandas as pd
from pathlib import Path

# Load CSV files
df_main = pd.read_csv("/dsi/gannot-lab/gannot-lab1/datasets/Librispeech_mfa/Test.csv", sep="|")
df_idx = pd.read_csv("/dsi/gannot-lab/gannot-lab1/users/mordehay/speech_repainting/exp/DiT_Anechoic_LibSp_conditional-masked-melspec_w-masked-pix=1/dit-net_dim768_depth18_heads12_dim-head64_dropout0.1_ff_mult2_T400_betaT0.02/repeat_all_freq-length=30_skip=75_cp=112000_mel_text=False_phoneme-without-space_g2p-no-nn_lm-weight=0p5_ctc-weight=0p1_bs=80/w1=1_w2=0.5_asr_start=320_mask=True/samples_info.csv", sep="|")

# Extract filenames for each sample index
out_df = pd.DataFrame({
    "Sample": df_idx["Sample"],
    "filename": [Path(df_main.iloc[s]["wav_path"]).stem for s in df_idx["Sample"]]
})

# Save results
out_df.to_csv("sample_filenames.csv", index=False)

print("Saved to sample_filenames.csv")