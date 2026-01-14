

import csv
import json
from pathlib import Path

# Input CSV and output JSON paths
csv_file = "/home/dsi/moradim/SpeechRepainting/all_experiments_with_hifi.csv"
output_json = "output.json"

# Mapping from length value in "experiment" to JSON key
length_map = {
    "-length=10": "0.1sec",
    "-length=20": "0.2sec",
    "-length=30": "0.3sec"
}

# Result dictionary with "without-text" as the main key
result = {"without-text": {key: [] for key in length_map.values()}}

with open(csv_file, newline='', encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        experiment = row["experiment"]
        sample_id = row["Sample"]
        filename = row["filename"]
        true_text = row["True text"]
        text4phonemes = row["text4phoneme"]
        trans_hifi_gan = row["trans_hifi_gan"]

        # Determine which duration this belongs to
        time_key = None
        for length_str, duration in length_map.items():
            if length_str in experiment:
                time_key = duration
                break
        if time_key is None:
            continue  # Skip if no matching length

        base_folder = Path(f"data/without-given-text/{duration}/sample_{sample_id}")

        # Create the JSON object for this sample
        entry = {
            "title": f"librispeech_{filename}",
            "id": f"example{sample_id}-{duration}",
            "maskedAsrLmText": text4phonemes.capitalize(),
            "inpaintedText": trans_hifi_gan.capitalize(),
            "targetText": true_text.capitalize(),
            "maskedAudio": str(base_folder / "masked_audio_time.wav"),
            "inpaintedAudio": str(base_folder / "generated_audio_hifi_gan.wav"),
            "targetAudio": str(base_folder / "gt_audio_hifi_gan.wav"),
            "maskedSpectrogram": str(base_folder / "masked_spec_image.png"),
            "inpaintedSpectrogram": str(base_folder / "generated_spec_image.png"),
            "targetSpectrogram": str(base_folder / "gt_spec_image.png"),
        }

        result["without-text"][time_key].append(entry)

# Save JSON
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"JSON saved to {output_json}")
