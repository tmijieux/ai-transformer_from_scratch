import numpy as np
from pathlib import Path

import torch
import torch.nn as nn



from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataset
from model import get_model
from config import Config, get_weights_file_path, get_config

from infer import greedy_decode



def run_validation(
    model,
    validation_dataset,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    device,
    print_msg,
    global_step,
    writer,
    num_examples=2
):
    count = 0
    for batch in validation_dataset:
        count += 1
        encoder_input = batch["encoder_input"].to(device)
        encoder_mask = batch["encoder_mask"].to(device)

        assert encoder_input.size(0) == 1 , "batch size must be 1 for validation"

        model_out = greedy_decode(
            model,
            encoder_input,
            encoder_mask,
            tokenizer_src,
            tokenizer_tgt,
            max_len,
            device
        )

        source_text = batch["src_text"][0]
        expected_text = batch["tgt_text"][0]
        model_out_text = tokenizer_tgt.decode(
            model_out.detach().cpu().numpy()
        )

        # source_texts.append(source_texts)
        # expected.append(target_text)
        # predicted.append(model_out_text)

        console_width = 20
        print_msg('-'*console_width)
        print_msg(f"SOURCE {source_text}")
        print_msg(f"EXPECTED {expected_text}")
        print_msg(f"PREDICTED {model_out_text}")

        if count == num_examples:
            break


def train_model(config: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("using device = ", device)

    Path(config.model_folder).mkdir(parents=True,exist_ok=True)

    #train_dataloader, validation_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    ds = get_dataset(config)
    model = get_model(
        config,
        ds.tokenizer_src.get_vocab_size(),
        ds.tokenizer_tgt.get_vocab_size(),
    ).to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nb_params = sum([np.prod(p.size()) for p in model_parameters])
    print("nb_parameters=", nb_params)

    writer = SummaryWriter(config.experiment_name)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        eps=1e-9,
    )

    initial_epoch = 0
    global_step = 0
    if config.preload is not None:
        model_filename = get_weights_file_path(config, config.preload)
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        initial_epoch = state["epoch"] + 1
        global_step = state["global_step"]
    else:
        print("no model to preload, starting from scratch")

    # model = torch.compile(model)
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=ds.tokenizer_src.token_to_id("[padding]"),
        label_smoothing=0.1
    ).to(device)

    for epoch in range(initial_epoch, config.num_epochs):
        model.train()
        batch_iterator = tqdm(
            ds.train,
            desc=f"Processing epoch {epoch:02d}"
        )
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device) # (batch, seq_len)
            decoder_input = batch["decoder_input"].to(device) # (batch, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (batch, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(device) # (batch, 1, seq_len, seq_len)

            encoder_output = model.encode(encoder_input, encoder_mask) # (batch, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)  # (batch, seq_len, d_model)

            proj_output  = model.project(decoder_output) #(batch, seq_len, tgt_vocab_size)

            label = batch["label"].to(device) # (batch, seq_len)

            # (batch, seq_len, tgt_vocab_size) --> (batch * seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, ds.tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            # backpropagate the loss
            loss.backward()

            # update the weight
            optimizer.step()
            optimizer.zero_grad()

            if global_step % 150 == 0:
                model.eval()
                with torch.no_grad():
                    run_validation(
                        model,
                        ds.validation,
                        ds.tokenizer_src,
                        ds.tokenizer_tgt,
                        config.seq_len,
                        device,
                        lambda msg: batch_iterator.write(msg),
                        global_step,
                        writer,
                    )
                model.train()

            global_step += 1

        # save the model
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step
        }, model_filename)

if __name__ == "__main__":
    train_model(get_config())

