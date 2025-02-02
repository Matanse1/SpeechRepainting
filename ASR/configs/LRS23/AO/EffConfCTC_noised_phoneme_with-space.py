# Imports
import ASR.nnet as nnet
# import nnet as nnet
import torch
import json

#tokenizer
tokenizer_path = '/home/dsi/moradim/SpeechRepainting/phoneme_to_number.json'
with open(tokenizer_path, 'r') as f:
    phoneme_to_number_loaded = json.load(f)
    for key in phoneme_to_number_loaded.keys():
        phoneme_to_number_loaded[key] = phoneme_to_number_loaded[key] + 1 #blank is zero so we need to add one
        
num_to_phoneme = {v: k for k, v in phoneme_to_number_loaded.items()}
num_to_phoneme[0] = 'blank'
vocab_char_map = phoneme_to_number_loaded
vocab_size = len(phoneme_to_number_loaded)
# Architecture

interctc_blocks = []
loss_weights = None
att_type = "patch"
strides_subsampling = 1 # we need the input length to be greater than the output length we set stide to 1
print(f"strides_subsampling: {strides_subsampling}")
# Training
batch_size = 8 #32
accumulated_steps = 4
grad_max_norm= None
eval_training = False
precision = torch.float32
recompute_metrics = True
callback_path = "/dsi/gannot-lab1/users/mordehay/phoneme_guidance_EffConfCTC_with-space" # where to save logs and model checkpoints

# Beam Search
beam_search = False
tokenizer_path = "ASR/media/tokenizerbpe256.model"
ngram_path = "ASR/media/6gram_lrs23.arpa"
ngram_offset = 100
beam_size = 16
ngram_alpha = 0.6
ngram_beta = 1.0
ngram_tmp = 1.0
neural_config_path = "ASR/configs/LRS23/LM/GPT-Small-demo.py"
neural_checkpoint = "checkpoints_epoch_10_step_2860.ckpt"
neural_alpha = 0.6
neural_beta = 1.0

custom_tokenizer = True
# Model
model = nnet.AudioEfficientConformerInterCTC(vocab_size=vocab_size, att_type=att_type, interctc_blocks=interctc_blocks, strides_subsampling=strides_subsampling)
model.compile(
    losses=nnet.CTCLoss(zero_infinity=True, assert_shorter=False),
    metrics=nnet.PhonemeErrorRate(),
    decoders=nnet.CTCGreedySearchDecoder(tokenizer_path=tokenizer_path, custom_tokenizer=custom_tokenizer, num_to_phoneme=num_to_phoneme)
    if not beam_search
    else nnet.CTCBeamSearchDecoder(
        tokenizer_path=tokenizer_path,
        beam_size=beam_size,
        ngram_path=ngram_path,
        ngram_tmp=ngram_tmp,
        ngram_alpha=ngram_alpha,
        ngram_beta=ngram_beta,
        ngram_offset=ngram_offset,
        neural_config_path=neural_config_path,
        neural_checkpoint=neural_checkpoint,
        neural_alpha=neural_alpha,
        neural_beta=neural_beta,
    ),
    loss_weights=loss_weights,
)

# Datasets
audio_max_length = 17 * 16000
collate_fn = nnet.CollateFn(
    inputs_params=[{"axis": 0, "padding": True}, {"axis": 2}],
    targets_params=({"axis": 1, "padding": True}, {"axis": 3}),
)

training_dataset = nnet.datasets.LibriSpeechPhoneme(
                    split="train",
                    sampling_rate=16000,
                    csv_loc="/dsi/gannot-lab1/datasets/Librispeech_mfa/",
                    base_data_dir="/dsi/gannot-lab1/datasets/Librispeech_mfa/", 
                    batch_size=batch_size,
                    collate_fn=collate_fn,
                    vocab_char_map=vocab_char_map,
                    use_input_text='phoneme'
                    )

evaluation_dataset = [
    nnet.datasets.LibriSpeechPhoneme(
                    split="test",
                    sampling_rate=16000,
                    csv_loc="/dsi/gannot-lab1/datasets/Librispeech_mfa/",
                    base_data_dir="/dsi/gannot-lab1/datasets/Librispeech_mfa/", 
                    batch_size=batch_size,
                    collate_fn=collate_fn,
                    vocab_char_map=vocab_char_map,
                    use_input_text='phoneme'
                    )
]
