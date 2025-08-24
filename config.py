from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    batch_size = 3
    num_epochs = 20
    learning_rate = 10**-4
    seq_len = 484
    d_model = 256
    lang_src = "en"
    lang_tgt = "fr"
    model_folder = "weights"
    model_basename = "tmodel_"
    #preload = "05"
    preload = None
    tokenizer_file = "tokenizers/tokenizer_{0}.json"
    experiment_name = "runs/tmodel"


def get_config() -> Config:
    return Config()


def get_weights_file_path(config: Config, epoch: str):
    model_folder = config.model_folder
    model_basename = config.model_basename
    model_filename = f"{model_basename}{epoch}.pt"

    return str(Path(".") / model_folder / model_filename)
