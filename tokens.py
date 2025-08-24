from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from config import Config

def get_all_sentences(dataset: Iterable[dict], lang: str):
    for item in dataset:
        yield item["translation"][lang]


def get_or_build_tokenizer(config: Config, dataset, lang: str):

    tokenizer_path = Path(config.tokenizer_file.format(lang))

    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[unknown]'))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(
            special_tokens=["[unknown]", "[padding]", "[start-of-sentence]", "[end-of-sentence]"],
            min_frequency=2,
        )
        tokenizer.train_from_iterator(
            get_all_sentences(dataset, lang),
            trainer=trainer,
        )

        tokenizer.save(str(tokenizer_path))

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer


@dataclass
class Tokenizers:
    source: Tokenizer
    target: Tokenizer

def get_or_build_tokenizers(config: Config, dataset_raw) -> Tokenizers:

    tokenizer_src = get_or_build_tokenizer(config, dataset_raw, config.lang_src)
    tokenizer_tgt = get_or_build_tokenizer(config, dataset_raw, config.lang_tgt)
    return Tokenizers(
        source=tokenizer_src,
        target=tokenizer_tgt,
    )
