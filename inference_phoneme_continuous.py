"""Phoneme-only inference helpers and CLI.

This module contains utilities to locate/load the trained phoneme classifier and
to build CTC targets from phoneme/text inputs. It is intended to be imported
by the full mel inference script, and can also be run standalone to sanity-check
the phoneme model and tokenization.
"""
import glob
import json
import os
import argparse

import torch

import ASR.nnet as nnet
from SDE import VPSDE


DEFAULT_PHONEME_CLASSIFIER_DIR = (
    "/dsi/gannot-lab/gannot-lab1/users/Alon_Matan/phoneme_classifier/"
    "phoneme_guidance_EffConfCTC_without-space"
)


def resolve_existing_path(path, fallback_paths=None, must_exist=True):
    fallback_paths = fallback_paths or []
    candidates = []
    if path is not None:
        candidates.append(os.path.expanduser(str(path)))
    candidates += [os.path.expanduser(str(p)) for p in fallback_paths if p is not None]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    if must_exist:
        raise FileNotFoundError("Could not find any of these paths:\n" + "\n".join(candidates))
    return candidates[0] if candidates else None


def find_latest_phoneme_checkpoint(phoneme_classifier_dir):
    pattern = os.path.join(phoneme_classifier_dir, "checkpoints_epoch_*_step_*.ckpt")
    ckpts = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not ckpts:
        raise FileNotFoundError(f"No phoneme classifier checkpoints found with pattern: {pattern}")
    return ckpts[0]


def load_phoneme_map(phoneme_map_path=None, remove_space=False):
    phoneme_map_path = resolve_existing_path(
        phoneme_map_path or "phoneme_to_number.json",
        fallback_paths=[
            os.path.join(os.getcwd(), "phoneme_to_number.json"),
            "/home/dsi/sellama/SpeechRepainting/phoneme_to_number.json",
            "/home/dsi/moradim/SpeechRepainting/phoneme_to_number.json",
        ],
        must_exist=True,
    )

    with open(phoneme_map_path, "r") as f:
        raw_map = json.load(f)

    phoneme_to_number = {}
    for key, value in raw_map.items():
        phoneme_to_number[key] = int(value) + 1  # blank = 0

    num_to_phoneme = {v: k for k, v in phoneme_to_number.items()}
    num_to_phoneme[0] = "blank"
    return phoneme_to_number, num_to_phoneme, phoneme_map_path


def unwrap_singleton(value):
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value


def text_to_string(value):
    value = unwrap_singleton(value)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return str(value)


def maybe_g2p_text(text, phoneme_to_number, with_space=False):
    try:
        from g2p_en import G2p
    except Exception as exc:
        raise RuntimeError(
            "No phoneme sequence was provided, and g2p_en is not available. "
            "Either pass dataset input_text / phoneme4guidance, or install g2p_en."
        ) from exc

    text = text_to_string(text)
    g2p_model = G2p()
    raw_phonemes = g2p_model(text)
    valid = set(phoneme_to_number.keys())
    phonemes = []
    for p in raw_phonemes:
        if p == " ":
            if with_space and "space" in valid:
                phonemes.append("space")
        elif p in valid:
            phonemes.append(p)
    return phonemes


def extract_phoneme_sequence(
    phoneme_to_number,
    guidance_text=None,
    input_text=None,
    phoneme4guidance=None,
    type_input_guidance="phoneme",
    with_space=False,
    remove_space=True,
):
    seq = None

    if phoneme4guidance is not None and phoneme4guidance != ["None"]:
        seq = unwrap_singleton(phoneme4guidance)

    if seq is None and input_text is not None:
        seq = unwrap_singleton(input_text)

    if seq is None and guidance_text is not None:
        seq = maybe_g2p_text(guidance_text, phoneme_to_number, with_space=with_space)

    if seq is None:
        raise RuntimeError(
            "Could not build phoneme guidance targets. Expected phoneme4guidance, input_text, or guidance_text."
        )

    if isinstance(seq, str):
        seq = seq.strip().split()

    clean_seq = []
    for item in seq:
        if item == "space" and remove_space:
            continue
        if item in phoneme_to_number:
            clean_seq.append(item)

    if len(clean_seq) == 0:
        raise RuntimeError(f"Empty phoneme target sequence after filtering. Original sequence: {seq}")

    return clean_seq


def phoneme_sequence_to_ctc_targets(phoneme_seq, phoneme_to_number, device):
    ids = [phoneme_to_number[p] for p in phoneme_seq]
    tokens = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    token_lengths = torch.tensor([tokens.shape[1]], dtype=torch.long, device=device)
    return tokens, token_lengths


def load_trained_phoneme_classifier(
    checkpoint_path,
    device,
    vocab_size,
    beta_min=0.1,
    beta_max=20.0,
    sde_N=1000,
    att_type="patch",
    interctc_blocks=None,
    strides_subsampling=1,
):
    if interctc_blocks is None:
        interctc_blocks = []

    model = nnet.AudioEfficientConformerInterCTC(
        vocab_size=vocab_size,
        att_type=att_type,
        interctc_blocks=interctc_blocks,
        strides_subsampling=strides_subsampling,
    )

    model.sde = VPSDE(beta_min=beta_min, beta_max=beta_max, N=sde_N, sampler_type="pc")

    model.compile(
        losses=nnet.CTCLoss(zero_infinity=True, assert_shorter=False),
        metrics=None,
        decoders=None,
        loss_weights=None,
    )

    model = model.to(device)
    model.device = device
    model.load(checkpoint_path, load_optimizer=False, strict=False)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    print(f"Loaded trained phoneme classifier from: {checkpoint_path}")
    return model


def continuous_t_to_classifier_diffusion_steps(t, phoneme_classifier):
    B = t.shape[0]
    sde_T = float(phoneme_classifier.sde.T)
    sde_N = int(phoneme_classifier.sde.N)
    sde_t = t / sde_T
    return (sde_t * (sde_N - 1)).view(B, 1, 1)


def write_phoneme_comparison_html(
    expected_phoneme_seq,
    predicted_phoneme_seq,
    output_path,
    mask=None,
    diffusion_step=None,
    total_steps=None,
    hop_length=160,
    sample_rate=16000,
):
    expected = expected_phoneme_seq or []
    predicted = predicted_phoneme_seq or []

    if isinstance(expected, str):
        expected = expected.strip().split()
    if isinstance(predicted, str):
        predicted = predicted.strip().split()

    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().tolist()
        if isinstance(mask, list) and len(mask) > 0 and isinstance(mask[0], list):
            mask = mask[0]
    else:
        mask = [1] * max(len(expected), len(predicted))

    def colorize_html(symbol, color):
        return f"<span style='color:red'>{symbol}</span>" if color == 0 else symbol

    rows = []
    max_len = max(len(expected), len(predicted))
    for i in range(max_len):
        true_ph = expected[i] if i < len(expected) else ""
        est_ph = predicted[i] if i < len(predicted) else ""
        color = mask[i] if i < len(mask) else 1
        true_colored = colorize_html(true_ph, color)
        est_colored = colorize_html(est_ph, color)
        time_sec = i * hop_length / sample_rate
        rows.append(f"<tr><td>{true_colored}</td><td>{est_colored}</td><td>{time_sec:.3f}</td></tr>")

    with open(output_path, "w") as f:
        f.write("<html><body>\n")
        title = "Phoneme comparison"
        if diffusion_step is not None and total_steps is not None:
            title += f" - step {diffusion_step}/{total_steps}"
        f.write(f"<h1 style='text-align: center;'>{title}</h1>\n")
        f.write("<table border='1' style='border-collapse: collapse; text-align: center;'>\n")
        f.write("<tr><th>expected</th><th>predicted</th><th>time[s]</th></tr>\n")
        f.write("\n".join(rows))
        f.write("</table>\n")
        f.write("</body></html>\n")

    print(f"Wrote phoneme comparison HTML to: {output_path}")


def asr_start_to_continuous_t(asr_start, sde, phoneme_classifier=None):
    if asr_start is None:
        return None
    if isinstance(asr_start, (list, tuple)):
        asr_start = asr_start[0]
    asr_start = float(asr_start)

    sde_T = float(sde.T)
    if asr_start <= sde_T:
        return asr_start

    if phoneme_classifier is not None and hasattr(phoneme_classifier, "sde"):
        N = int(phoneme_classifier.sde.N)
        T = float(phoneme_classifier.sde.T)
    else:
        N = int(sde.N)
        T = sde_T

    return (asr_start / max(N - 1, 1)) * T


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phoneme_map", default=None)
    parser.add_argument("--phoneme_dir", default=None)
    parser.add_argument("--phoneme_ckpt", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected", default=None, help="Expected phoneme sequence or guidance text")
    parser.add_argument("--predicted", default=None, help="Predicted / real phoneme sequence for comparison")
    parser.add_argument("--output_html", default="phoneme_comparison.html", help="Path to write comparison HTML")
    args = parser.parse_args()  

    phoneme_to_number, num_to_phoneme, map_path = load_phoneme_map(args.phoneme_map)
    print("Loaded phoneme map:", map_path)

    if args.phoneme_ckpt is not None or args.phoneme_dir is not None:
        ckpt = args.phoneme_ckpt
        if ckpt is None:
            ckpt_dir = resolve_existing_path(args.phoneme_dir, fallback_paths=[DEFAULT_PHONEME_CLASSIFIER_DIR])
            ckpt = find_latest_phoneme_checkpoint(ckpt_dir)
        print("Using phoneme checkpoint:", ckpt)
        model = load_trained_phoneme_classifier(
            checkpoint_path=ckpt,
            device=args.device,
            vocab_size=len(phoneme_to_number) + 1,
        )
    else:
        model = None

    if args.expected is not None or args.predicted is not None:
        expected_seq = None
        predicted_seq = None

        if args.expected is not None:
            expected_seq = extract_phoneme_sequence(
                phoneme_to_number,
                guidance_text=args.expected,
                type_input_guidance="phoneme",
                with_space=False,
            )
            print("Expected phoneme sequence:", expected_seq)

        if args.predicted is not None:
            predicted_seq = extract_phoneme_sequence(
                phoneme_to_number,
                guidance_text=args.predicted,
                type_input_guidance="phoneme",
                with_space=False,
            )
            print("Predicted phoneme sequence:", predicted_seq)

        if expected_seq is not None and predicted_seq is not None:
            write_phoneme_comparison_html(
                expected_phoneme_seq=expected_seq,
                predicted_phoneme_seq=predicted_seq,
                output_path=args.output_html,
                mask=None,
                diffusion_step=None,
                total_steps=None,
                hop_length=160,
                sample_rate=16000,
            )
        else:
            print("Both --expected and --predicted are required to generate HTML comparison.")

if __name__ == "__main__":
    main()
