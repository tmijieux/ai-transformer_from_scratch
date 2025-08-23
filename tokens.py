from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from config import Config
from dataset import get_all_sentences

def build_tokenizer(config: Config, dataset, lang: str):

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
