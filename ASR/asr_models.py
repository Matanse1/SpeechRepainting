from .nnet import AudioEfficientConformerInterCTC
from .nnet import CTCLoss, CTCBeamSearchDecoder
import sentencepiece as spm

def get_models(dataset):
    assert dataset.lower() in ['lrs3', 'lrs2'], 'Dataset must be LRS3 or LRS2'
    asr_guidance_net = AudioEfficientConformerInterCTC(interctc_blocks=[], T=400, beta_0=0.0001, beta_T=0.02)
    checkpoint_ao = f"/dsi/gannot-lab1/users/mordehay/asr_yochai_lipvoicer/checkpoints_ft_{dataset.lower()}.ckpt"
    asr_guidance_net.compile(losses=CTCLoss(zero_infinity=True, assert_shorter=False), loss_weights=None)
    asr_guidance_net = asr_guidance_net.cuda()
    asr_guidance_net.load(checkpoint_ao)
    asr_guidance_net.eval()
    tokenizer_path = "/dsi/gannot-lab1/users/mordehay/asr_yochai_lipvoicer/tokenizerbpe256.model" #thi is the tokenizer
    tokenizer = spm.SentencePieceProcessor(tokenizer_path)  # for converting text to tokens
    ngram_path = "/dsi/gannot-lab1/users/mordehay/asr_yochai_lipvoicer/6gram_lrs23.arpa" # this is the language model
    neural_config_path = "ASR/configs/LRS23/LM/GPT-Small-demo.py" # this is the acoustic model
    neural_checkpoint = "/dsi/gannot-lab1/users/mordehay/asr_yochai_lipvoicer/checkpoints_epoch_10_step_2860.ckpt" # this is the acoustic model's weights

    # Decoder for converting tokens to text at the ASR output - With LM
    # decoder = CTCBeamSearchDecoder( 
    #     tokenizer_path=tokenizer_path,
    #     ngram_path=ngram_path,
    #     neural_config_path=neural_config_path,
    #     neural_checkpoint=neural_checkpoint,
    # ) 
        # Decoder for converting tokens to text at the ASR output - Without LM
    decoder = CTCBeamSearchDecoder( 
        ngram_alpha=0,
        tokenizer_path=tokenizer_path,
        ngram_path=ngram_path,
        neural_config_path=None ,#neural_config_path,
        neural_checkpoint=neural_checkpoint,
    ) 

    return asr_guidance_net, tokenizer, decoder

def get_models_without_noise(dataset):
    assert dataset.lower() in ['lrs3', 'lrs2'], 'Dataset must be LRS3 or LRS2'
    asr_guidance_net = AudioEfficientConformerInterCTC(interctc_blocks=[], T=400, beta_0=0.0001, beta_T=0.02)
    checkpoint_ao = f"ASR/callbacks/LRS23/AO/EffConfCTC/checkpoints_ft_{dataset.lower()}.ckpt"
    asr_guidance_net.compile(losses=CTCLoss(zero_infinity=True, assert_shorter=False), loss_weights=None)
    asr_guidance_net = asr_guidance_net.cuda()
    asr_guidance_net.load(checkpoint_ao)
    asr_guidance_net.eval()
    tokenizer_path = "ASR/media/tokenizerbpe256.model"
    tokenizer = spm.SentencePieceProcessor(tokenizer_path)  # for converting text to tokens
    ngram_path = "ASR/media/6gram_lrs23.arpa"
    neural_config_path = "ASR/configs/LRS23/LM/GPT-Small-demo.py"
    neural_checkpoint = "checkpoints_epoch_10_step_2860.ckpt"

    # Decoder for converting tokens to text at the ASR output
    decoder = CTCBeamSearchDecoder( 
        tokenizer_path=tokenizer_path,
        ngram_path=ngram_path,
        neural_config_path=neural_config_path,
        neural_checkpoint=neural_checkpoint,
    ) 

    return asr_guidance_net, tokenizer, decoder