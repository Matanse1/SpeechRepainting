from pathlib import Path
import shutil

# Define paths
root_dir = Path("/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/DiT_Anechoic_LibSp_conditional-masked-melspec_w-masked-pix=1/dit-net_dim768_depth18_heads12_dim-head64_dropout0.1_ff_mult2_T400_betaT0.02/repeat_all_freq-length=100_skip=150_cp=112000_mel_text=True_no-guidance")  # Adjust if needed
output_dir = Path("/dsi/gannot-lab1/users/mordehay/speech_repainting/exp/DiT_Anechoic_LibSp_conditional-masked-melspec_w-masked-pix=1/dit-net_dim768_depth18_heads12_dim-head64_dropout0.1_ff_mult2_T400_betaT0.02/masked_audio_collection_repeat_all_freq-length=100_skip=150")
output_dir.mkdir(exist_ok=True)  # Create the new folder

# Collect and copy the masked_audio_time.wav files
files_copied = []
for sample_folder in root_dir.glob("sample_*"):
    masked_audio_path = sample_folder / "masked_audio_time.wav"
    if masked_audio_path.exists():
        dest_path = output_dir / f"{sample_folder.name}_masked_audio_time.wav"
        shutil.copy(masked_audio_path, dest_path)
        files_copied.append(dest_path)

# Create the manifest file
manifest_path = output_dir / "manifest.txt"
with open(manifest_path, "w") as f:
    f.write(f"{output_dir}\n")  # Root directory
    for file in files_copied:
        f.write(f"{file.name}\t{file.stat().st_size}\n")

print(f"Copied {len(files_copied)} files and created {manifest_path}")
