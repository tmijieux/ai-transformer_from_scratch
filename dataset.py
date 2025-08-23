import torch
import torch.nn as nn
from torch.utils.data import Dataset

def make_input_mask(encoder_input, padding_token):
    return (encoder_input != padding_token).unsqueeze(0).unsqueeze(0).int()

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0

def make_decoder_mask(decoder_input, padding_token):
    return make_input_mask(decoder_input, padding_token) & causal_mask(decoder_input.size(0))

class BilingualDataset(Dataset):
    def __init__(
            self,
            dataset,
            tokenizer_src,
            tokenizer_tgt,
            src_lang:str,
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


