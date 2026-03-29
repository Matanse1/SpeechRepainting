import pandas as pd
from pathlib import Path
import re



import pandas as pd
from pathlib import Path
import re


# Load metrics file
metrics_df = pd.read_csv("/home/dsi/moradim/SpeechRepainting/metric_results_dit_mel-text=False_g2p-no-nn_WER_length=10+20+30_skip=25+50+75_lm=0p5_ctc=0p1_bs=80.csv", sep="|")  # use the actual path to your metrics file
# Extract Sample number from 'sample_path'
metrics_df["Sample"] = metrics_df["sample_path"].apply(lambda p: int(re.search(r"sample_(\d+)", p).group(1)))
# Extract experiment name (-length=..._skip=...)
metrics_df["experiment"] = metrics_df["sample_path"].apply(lambda p: re.search(r"-length=\d+_skip=\d+", p).group(0))

# Mapping CSV from earlier step
mapping_df = pd.read_csv("/home/dsi/moradim/SpeechRepainting/sample_filenames.csv")

# Your base directories
base_dirs = [
    Path("/dsi/gannot-lab/gannot-lab1/users/mordehay/speech_repainting/exp/DiT_Anechoic_LibSp_conditional-masked-melspec_w-masked-pix=1/dit-net_dim768_depth18_heads12_dim-head64_dropout0.1_ff_mult2_T400_betaT0.02/repeat_all_freq-length=30_skip=75_cp=112000_mel_text=False_phoneme-without-space_g2p-no-nn_lm-weight=0p5_ctc-weight=0p1_bs=80/w1=1_w2=0.5_asr_start=320_mask=True"),
    Path("/dsi/gannot-lab/gannot-lab1/users/mordehay/speech_repainting/exp/DiT_Anechoic_LibSp_conditional-masked-melspec_w-masked-pix=1/dit-net_dim768_depth18_heads12_dim-head64_dropout0.1_ff_mult2_T400_betaT0.02/repeat_all_freq-length=20_skip=50_cp=112000_mel_text=False_phoneme-without-space_g2p-no-nn_lm-weight=0p5_ctc-weight=0p1_bs=80/w1=1_w2=0.5_asr_start=320_mask=True"),
    Path("/dsi/gannot-lab/gannot-lab1/users/mordehay/speech_repainting/exp/DiT_Anechoic_LibSp_conditional-masked-melspec_w-masked-pix=1/dit-net_dim768_depth18_heads12_dim-head64_dropout0.1_ff_mult2_T400_betaT0.02/repeat_all_freq-length=10_skip=25_cp=112000_mel_text=False_phoneme-without-space_g2p-no-nn_lm-weight=0p5_ctc-weight=0p1_bs=80/w1=1_w2=0.5_asr_start=320_mask=True"),
]
asr_rows = []

# Loop over all ASR folders
for base_dir in base_dirs:
    exp_name = re.search(r"-length=\d+_skip=\d+", str(base_dir)).group(0)
    folder = base_dir 

    for _, row in mapping_df.iterrows():
        sample_num = row["Sample"]
        filename = row["filename"]

        txt_file = folder / f"sample_{sample_num}" / "asr_text.txt"
        if txt_file.exists():
            with open(txt_file, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
            true_text = lines[0].split("True text:", 1)[1].strip()
            text4phoneme = lines[1].split("text4phoneme:", 1)[1].strip()

            asr_rows.append({
                "experiment": exp_name,
                "Sample": sample_num,
                "filename": filename,
                "True text": true_text,
                "text4phoneme": text4phoneme
            })

# Create ASR DataFrame
asr_df = pd.DataFrame(asr_rows)

# Merge ASR + metrics
merged_df = pd.merge(asr_df, metrics_df[["Sample", "experiment", "trans_hifi_gan"]], on=["Sample", "experiment"], how="left")

# Save
merged_df.to_csv("all_experiments_with_hifi.csv", index=False)
print("Saved to all_experiments_with_hifi.csv")
