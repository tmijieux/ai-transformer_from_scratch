print("importing torch...")
from dataclasses import dataclass
from tokenizers import Tokenizer
import torch

print("\n\nimporting local code...")
from config import get_weights_file_path, get_config
from model import Transformer, get_model
from dataset import make_input_mask, causal_mask


print("\n\nfinished imports!\n\n")


def remove_compiled_state(q):
    for k,v in q.items():
        if k.startswith("_orig_mod."):
            break
    else:
        return q

    from collections import OrderedDict
    res = OrderedDict()
    for k,v in q.items():
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
            res[k] = v
    return res


@dataclass
class ModelAndTokenizers:
    model: Transformer
    tokenizer_src: Tokenizer
    tokenizer_tgt: Tokenizer
    device: torch.device

def load_model() -> ModelAndTokenizers:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_ok = False
    if torch.cuda.is_available():
        device_cap = torch.cuda.get_device_capability()
        if device_cap in ((7, 0), (8, 0), (9, 0)):
            gpu_ok = True
        else:
            print("device_cap=",device_cap)

    if not gpu_ok:
        print(
            "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower "
            "than expected."
        )

    print("using device = ", device)
    config = get_config()

    from tokens import get_or_build_tokenizer

    tokenizer_src = get_or_build_tokenizer(config, None, config.lang_src)
    tokenizer_tgt = get_or_build_tokenizer(config, None, config.lang_tgt)
    print("src.vocab_size=", tokenizer_src.get_vocab_size())
    print("tgt.vocab_size=", tokenizer_tgt.get_vocab_size())

    model = get_model(
        config,
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
    ).to(device)

    if config.preload is not None:
        model_filename = get_weights_file_path(config, config.preload)

        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        q = state["model_state_dict"]
        q = remove_compiled_state(q)
        #print("q=",q)
        model.load_state_dict(q)
    else:
        raise Exception("could not load model - no preload")

    print("compiling model")
    model: Transformer = torch.compile(model)
    model.eval() # evaluation mode (disable dropout - some norms?)
    print("model compiled!")

    return ModelAndTokenizers(
        model=model, 
        tokenizer_src=tokenizer_src, 
        tokenizer_tgt=tokenizer_tgt, 
        device=device,
    )

def build_input_from_string(tokenizer_src: Tokenizer, input_text: str, seq_len: int):
    enc_input_tokens = tokenizer_src.encode(input_text).ids
    enc_num_padding_tokens = seq_len - len(enc_input_tokens) - 2 # s-o-s / e-o-s
    sos = torch.tensor(
        [tokenizer_src.token_to_id('[start-of-sentence]')],
        dtype=torch.int64,
    )
    eos = torch.tensor(
        [tokenizer_src.token_to_id('[end-of-sentence]')],
        dtype=torch.int64,
    )
    padding = torch.tensor(
        [tokenizer_src.token_to_id('[padding]')],
        dtype=torch.int64,
    )
    enc_input = torch.cat(
        [
            sos,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            eos,
            torch.tensor([padding] * enc_num_padding_tokens, dtype=torch.int64)
        ]
    )
    enc_mask = make_input_mask(enc_input, padding)
    return enc_input, enc_mask

def as_single_batch(input_sequence: torch.Tensor):
    return input_sequence.unsqueeze(dim = 0)

def greedy_decode(
    m: ModelAndTokenizers,
    source: torch.Tensor,
    source_mask: torch.Tensor,
    max_len: int
):
    start_of_sentence_idx: int = m.tokenizer_tgt.token_to_id("[start-of-sentence]")
    end_of_sentence_idx: int = m.tokenizer_tgt.token_to_id("[end-of-sentence]")

    # precompute the encoder output and resute it for every token we get from the decoder
    encoder_output = m.model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(start_of_sentence_idx).type_as(source).to(m.device)

    while True:
        if decoder_input.size(1) == max_len:
            print("broke because max_len")
            break
        # build mask for the target (decoder input)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(m.device)

        # calculate the output
        out = m.model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # out: (batch_size(1), seq_len, embed_size)
        prob = m.model.project(out[:,-1,:])#-1 takes only last token of output sequence ?

        # (Batch, Seq_Len, embed_size)
        # --project-- > prob: (Batch, Seq_Len, vocab_size)
        
        # select the token with the max probability (because it is a greedy search)
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([
            decoder_input,
            torch.empty(1,1).type_as(source).fill_(next_word.item()).to(m.device)
        ], dim=1)

        if next_word == end_of_sentence_idx:
            print("broke because EOS")
            break
    return decoder_input.squeeze(0)


def infer(m: ModelAndTokenizers, input_text: str):

    print("input_text=", input_text)
    enc_input, enc_mask = build_input_from_string(
        m.tokenizer_src, input_text, m.model.src_params.seq_len
    )
    print("enc_input=", enc_input)
    model_out = greedy_decode(
        m,
        as_single_batch(enc_input.to(m.device)),
        as_single_batch(enc_mask.to(m.device)),
        m.model.tgt_params.seq_len,
    )
    model_out_text: str = m.tokenizer_tgt.decode(
        model_out.detach().cpu().numpy()
    )
    return model_out_text


def infer_repl():
    print("loading model...")
    model = load_model()
    # print("model=", model)

    print("\n READY ! \n")
    while True:
        try:
            input_text = input("\nEnter the english phrase to translate:\n> ")
            if input_text == "/bye" or input_text == "quit" or input_text == "exit":
                break
            result = infer(model, input_text)
            print("result:", result)
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    with torch.no_grad():
        infer_repl()
