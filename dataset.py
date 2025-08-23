import datasets
from dataclasses import dataclass
from typing import Iterable
from tokenizers import Tokenizer
import torch
import torch.utils.data
from torch.utils.data import DataLoader, random_split

from config import Config


@dataclass
class DataSet:
    train: DataLoader
    validation: DataLoader
    tokenizer_src: Tokenizer
    tokenizer_tgt: Tokenizer


def make_input_mask(encoder_input: torch.Tensor, padding_token: torch.Tensor):
    return (encoder_input != padding_token).unsqueeze(0).unsqueeze(0).int()

def causal_mask(size: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0

def make_decoder_mask(decoder_input: torch.Tensor, padding_token: torch.Tensor):
    return make_input_mask(decoder_input, padding_token) & causal_mask(decoder_input.size(0))

class BilingualDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        tokenizer_src: Tokenizer,
        tokenizer_tgt: Tokenizer,
        src_lang: str,
        tgt_lang: str,
        seq_len: int
    ):
        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.start_of_sentence_token = torch.tensor(
            [tokenizer_src.token_to_id('[start-of-sentence]')],
            dtype=torch.int64,
        )
        self.end_of_sentence_token = torch.tensor(
            [tokenizer_src.token_to_id('[end-of-sentence]')],
            dtype=torch.int64,
        )
        self.padding_token = torch.tensor(
            [tokenizer_src.token_to_id('[padding]')],
            dtype=torch.int64,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        src_target_pair = self.dataset[index]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # s-o-s / e-o-s
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # s-o-s

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("sentence is too long")

        encoder_input = torch.cat(
            [
                self.start_of_sentence_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.end_of_sentence_token,
                torch.tensor([self.padding_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        decoder_input = torch.cat(
            [
                self.start_of_sentence_token, # only s-o-s
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.padding_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.end_of_sentence_token, # only e-o-s to label (what expected as output)
                torch.tensor([self.padding_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, # (Seq_Len,)
            "decoder_input": decoder_input, # (Seq_Len,)
            "encoder_mask": make_input_mask(encoder_input, self.padding_token), # (1, 1, seq_len)
            "decoder_mask": make_decoder_mask(decoder_input, self.padding_token), # (1, seq_len,) & (1, seq_len, seq_len),
            "label":label, # (Seq_Len,)
            "src_text":src_text,
            "tgt_text":tgt_text,
        }





def get_all_sentences(dataset: Iterable[dict], lang: str):
    for item in dataset:
        yield item["translation"][lang]



def get_dataset(config: Config) -> DataSet:
    print("loading dataset")
    dataset_raw: datasets.Dataset = datasets.load_dataset(
        "Helsinki-NLP/opus_books",
        f"{config.lang_src}-{config.lang_tgt}", 
        split="train",
    )
    print("dataset loaded.")

    from tokens import build_tokenizer

    tokenizer_src = build_tokenizer(config, dataset_raw, config.lang_src)
    tokenizer_tgt = build_tokenizer(config, dataset_raw, config.lang_tgt)

    # keep 90% for training and 10% for validation
    train_dataset_size = int(0.9 * len(dataset_raw))
    validation_dataset_size = len(dataset_raw) - train_dataset_size

    train_dataset_raw, validation_dataset_raw = random_split(
        dataset_raw,
        [train_dataset_size, validation_dataset_size]
    )

    train_dataset = BilingualDataset(
        train_dataset_raw,
        tokenizer_src,
        tokenizer_tgt,
        config.lang_src,
        config.lang_tgt,
        config.seq_len,
    )

    validation_dataset = BilingualDataset(
        validation_dataset_raw,
        tokenizer_src,
        tokenizer_tgt,
        config.lang_src,
        config.lang_tgt,
        config.seq_len,
    )

    max_len_src = 0
    max_len_tgt = 0
    for item in dataset_raw:
        src_ids = tokenizer_src.encode(item["translation"][config.lang_src]).ids
        tgt_ids = tokenizer_tgt.encode(item["translation"][config.lang_tgt]).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"max length of source sentence {max_len_src}")
    print(f"max length of target sentence {max_len_tgt}")

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

    return DataSet(
        train=train_dataloader,
        validation=validation_dataloader, 
        tokenizer_src=tokenizer_src,
        tokenizer_tgt=tokenizer_tgt
    )

