from .nnet import AudioEfficientConformerInterCTC, PhonemeErrorRate
from .nnet import CTCLoss, CTCBeamSearchDecoder, CTCGreedySearchDecoder
import sentencepiece as spm
import json

def get_models(dataset, type_input_guidance='text', with_space=False):

        
    assert dataset.lower() in ['lrs3', 'lrs2'], 'Dataset must be LRS3 or LRS2'
    if type_input_guidance == 'text':
        print("The Guidance is ASR")
        asr_guidance_net = AudioEfficientConformerInterCTC(interctc_blocks=[], T=400, beta_0=0.0001, beta_T=0.02, strides_subsampling=2)
        checkpoint_ao = f"/dsi/gannot-lab1/users/mordehay/asr_yochai_lipvoicer/checkpoints_ft_{dataset.lower()}.ckpt"
        tokenizer_path = "/dsi/gannot-lab1/users/mordehay/asr_yochai_lipvoicer/tokenizerbpe256.model" #thi is the tokenizer
        tokenizer = spm.SentencePieceProcessor(tokenizer_path)  # for converting text to tokens
        
        ngram_path = "/dsi/gannot-lab1/users/mordehay/asr_yochai_lipvoicer/6gram_lrs23.arpa" # this is the language model
        neural_config_path = "ASR/configs/LRS23/LM/GPT-Small-demo.py" # this is the acoustic model
        neural_checkpoint = "/dsi/gannot-lab1/users/mordehay/asr_yochai_lipvoicer/checkpoints_epoch_10_step_2860.ckpt" # this is the acoustic model's weights
        # Decoder for converting tokens to text at the ASR output - Without LM
        decoder = CTCBeamSearchDecoder( 
        ngram_alpha=0,
        tokenizer_path=tokenizer_path,
        ngram_path=ngram_path,
        neural_config_path=None ,#neural_config_path,
        neural_checkpoint=neural_checkpoint,
                ) 
        metric = None
    elif type_input_guidance == 'phoneme':
        print("The Guidance is Phoneme")
        #tokenizer
        tokenizer_path = '/home/dsi/moradim/SpeechRepainting/phoneme_to_number.json'
        with open(tokenizer_path, 'r') as f:
            phoneme_to_number_loaded = json.load(f)
            for key in phoneme_to_number_loaded.keys():
                phoneme_to_number_loaded[key] = phoneme_to_number_loaded[key] + 1 #blank is zero so we need to add one
        num_to_phoneme = {v: k for k, v in phoneme_to_number_loaded.items()}
        num_to_phoneme[0] = 'blank'
        vocab_size = len(phoneme_to_number_loaded)
        tokenizer = phoneme_to_number_loaded
        asr_guidance_net = AudioEfficientConformerInterCTC(vocab_size=vocab_size, interctc_blocks=[], T=400, beta_0=0.0001, beta_T=0.02, strides_subsampling=1)
        # checkpoint_ao = "/dsi/gannot-lab1/users/mordehay/phoneme_guidance_EffConfCTC/checkpoints_epoch_9_step_9285.ckpt" no-space
        if with_space:
            checkpoint_ao = '/dsi/gannot-lab1/users/mordehay/phoneme_guidance_EffConfCTC_with-space-con/checkpoints_epoch_32_step_44024.ckpt' # with-space 
        else:
            checkpoint_ao = '/dsi/gannot-lab1/users/mordehay/phoneme_guidance_EffConfCTC_without-space/checkpoints_epoch_38_step_52278.ckpt'

        decoder = CTCGreedySearchDecoder(tokenizer_path=tokenizer_path, custom_tokenizer=True, num_to_phoneme=num_to_phoneme)
        metric = PhonemeErrorRate()
    
    asr_guidance_net.compile(losses=CTCLoss(zero_infinity=True, assert_shorter=False), loss_weights=None, metrics=metric, decoders=decoder)
    asr_guidance_net = asr_guidance_net.cuda()
    asr_guidance_net.load(checkpoint_ao)
    asr_guidance_net.eval()



    # Decoder for converting tokens to text at the ASR output - With LM
    # decoder = CTCBeamSearchDecoder( 
    #     tokenizer_path=tokenizer_path,
    #     ngram_path=ngram_path,
    #     neural_config_path=neural_config_path,
    #     neural_checkpoint=neural_checkpoint,
    # ) 


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