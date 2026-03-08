# from .sashimi import Sashimi
from .wavenet import WaveNet
from .u_net import Unet
from .dit import DiT
from .mel_spec_wavlm_phoneme_classifier import WavlmMelSpecPhonemeClassifier

def construct_model(model_cfg):
    name = model_cfg.pop("_name_")
    model_cls = {
        "wavenet": WaveNet,
        # "sashimi": Sashimi,
    }[name]
    model = model_cls(**model_cfg)
    model_cfg["_name_"] = name # restore
    return model


def model_identifier(model_cfg):
    model_cls = {
        "wavenet": WaveNet,
        "unet": Unet,
        "dit": DiT,
        "phoneme_classifier": WaveNet,
        "wavlm_phoneme_classifier_masked_speech": WavlmMelSpecPhonemeClassifier
    }[model_cfg._name_]
    return model_cls.name(model_cfg)
