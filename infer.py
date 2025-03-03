print("import torch")
import torch

print("import local code")

from config import get_weights_file_path, get_config
from model import get_model
from dataset import make_input_mask, causal_mask
from tokens import build_tokenizer


print("finished imports")

def load_model():
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
    tokenizer_src = build_tokenizer(config, None, config["lang_src"])
    tokenizer_tgt = build_tokenizer(config, None, config["lang_tgt"])

    print("src.vocab_size=",tokenizer_src.get_vocab_size())
    print("tgt.vocab_size=",tokenizer_tgt.get_vocab_size())

    model = get_model(
        config,
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
    ).to(device)

    if config["preload"] is not None:
        model_filename = get_weights_file_path(config, config["preload"])

        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
    else:
        raise Exception("could not load model - no preload")

    print("compiling model")
    model = torch.compile(model)
    model.eval() # evaluation mode (disable dropout - some norms?)
    print("model compiled!")

    return model, tokenizer_src, tokenizer_tgt, device

def build_input_from_string(tokenizer_src, input_text: str, seq_len: int):
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

def as_single_batch(input_sequence):
    return input_sequence.unsqueeze(dim = 0)

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):

    start_of_sentence_idx = tokenizer_tgt.token_to_id("[start-of-sentence]")
    end_of_sentence_idx = tokenizer_tgt.token_to_id("[end-of-sentence]")

    # precompute the encoder output and resute it for every token we get from the decoder

    encoder_output = model.encode(source, source_mask)

    decoder_input = torch.empty(1,1).fill_(start_of_sentence_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break
        # build mask for the target (decoder input)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate the output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        # out: (batch_size(1), seq_len, embed_size)


        prob = model.project(out[:,-1])#-1 takes only last token in seq ?
        # (Batch, Seq_Len, embed_size)
        # --project-- > prob: (Batch, Seq_Len, vocab_size)

        # select the token with the max probability (because it is a greedy search)
        _, next_word = torch.max(prob, dim=1)

        decoder_input = torch.cat([
            decoder_input,
            torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)
        ], dim=1)

        if next_word == end_of_sentence_idx:
            break
    return decoder_input.squeeze(0)


def infer(model_tokenizers, input_text: str):
    model, tokenizer_src, tokenizer_tgt, device = model_tokenizers

    enc_input, enc_mask = build_input_from_string(
        tokenizer_src, input_text, model.src_params.seq_len
    )

    model_out = greedy_decode(
        model,
        as_single_batch(enc_input.to(device)),
        as_single_batch(enc_mask.to(device)),
        tokenizer_src,
        tokenizer_tgt,
        model.tgt_params.seq_len,
        device
    )
    model_out_text = tokenizer_tgt.decode(
        model_out.detach().cpu().numpy()
    )
    return model_out_text


def infer_repl():
    print("loading...")
    model = load_model()
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
