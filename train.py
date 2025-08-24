from dataclasses import dataclass
import time
from typing import Callable
import typing
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import DataSet, MyBatch, get_dataset_raw, tokenize_dataset
from model import Transformer, get_model
from config import Config, get_weights_file_path, get_config

from infer import ModelAndTokenizers, greedy_decode
from tokens import Tokenizers, get_or_build_tokenizers
import gc



def run_validation(
    m: ModelAndTokenizers,
    validation_dataset: torch.utils.data.DataLoader,
    max_len: int,
    print_msg: Callable[[str],None],
    global_step: int,
    writer: SummaryWriter,
    num_examples: int = 2
):
    count = 0
    for batch in validation_dataset:
        count += 1
        encoder_input = batch["encoder_input"].to(m.device)
        encoder_mask = batch["encoder_mask"].to(m.device)

        assert encoder_input.size(0) == 1 , "batch size must be 1 for validation"

        model_out = greedy_decode(m, encoder_input, encoder_mask, max_len)

        source_text = batch["src_text"][0]
        expected_text = batch["tgt_text"][0]
        model_out_text = m.tokenizer_tgt.decode(
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


@dataclass
class Trainer:
    
    model: Transformer
    loss_fn: nn.CrossEntropyLoss
    writer: SummaryWriter
    optimizer: torch.optim.Optimizer


    device: torch.device
    cpu: torch.device


def find_best_batch_size(trainer: Trainer, dataset: DataSet):
    min_batchsize = -1
    min_duration = 99999999999999999

    #for batch_size in [32,64,128,256]:
    for batch_size in [1,2,3]:
        #dl = torch.utils.data.DataLoader(dataset.train_dataset, batch_size=batch_size, shuffle=True)
        dl = torch.utils.data.DataLoader(dataset.train_dataset, batch_size=batch_size)
        batch_iterator = tqdm(
            dl,
            desc=f"Trying out batchsize {batch_size}"
        )
        
        batch: MyBatch
        step = 0
        durations = []
        for batch in batch_iterator:
            start_time = time.perf_counter()
            do_one_batch(batch, batch_iterator, trainer, global_step=1)

            gc.collect()
            end_time = time.perf_counter()

            durations.append(end_time-start_time)
            step = step+1
            if step >= 100:
                break
        avg_duration = (sum(durations) / len(durations) ) / batch_size
        if avg_duration < min_duration:
            min_duration = avg_duration
            min_batchsize = batch_size
    

    print("best_batch_size=", min_batchsize)
    return min_batchsize



def do_one_batch(
    batch: MyBatch, 
    batch_iterator: tqdm, 
    trainer: Trainer,
    global_step: int,
):
    encoder_input = batch["encoder_input"].to(trainer.device) # (batch, seq_len)
    decoder_input = batch["decoder_input"].to(trainer.device) # (batch, seq_len)
    encoder_mask = batch["encoder_mask"].to(trainer.device) # (batch, 1, 1, seq_len)
    decoder_mask = batch["decoder_mask"].to(trainer.device) # (batch, 1, seq_len, seq_len)

    encoder_output = trainer.model.encode(encoder_input, encoder_mask) # (batch, seq_len, d_model)
    decoder_output = trainer.model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)  # (batch, seq_len, d_model)

    proj_output  = trainer.model.project(decoder_output) #(batch, seq_len, tgt_vocab_size)

    print("input_size=",encoder_input.element_size() * encoder_input.nelement())
    print("encoder_output=",encoder_output.element_size() * encoder_output.nelement())
    print("d_input_size=",decoder_output.element_size() * decoder_input.nelement())

    label = batch["label"].to(trainer.device) # (batch, seq_len)

    # (batch, seq_len, tgt_vocab_size) --> (batch * seq_len, tgt_vocab_size)
    loss = trainer.loss_fn(proj_output.view(-1, tokenizers.target.get_vocab_size()), label.view(-1))
    batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

    trainer.writer.add_scalar("train loss", loss.item(), global_step)
    trainer.writer.flush()

    # backpropagate the loss
    loss.backward()

    # update the weight
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()


    encoder_input.to(trainer.cpu)
    decoder_input.to(trainer.cpu)
    encoder_mask.to(trainer.cpu)
    decoder_mask.to(trainer.cpu)
    label.to(trainer.cpu)




def train_model(config: Config, dataset: DataSet, tokenizers: Tokenizers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device = ", device)
    Path(config.model_folder).mkdir(parents=True, exist_ok=True)
    model = get_model(
        config,
        tokenizers.source.get_vocab_size(),
        tokenizers.target.get_vocab_size(),
    ).to(device)

    

    m = ModelAndTokenizers(
        model=model, 
        tokenizer_src=tokenizers.source, 
        tokenizer_tgt=tokenizers.target, 
        device=device
    )

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
    global_step = 1
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
        ignore_index=tokenizers.source.token_to_id("[padding]"),
        label_smoothing=0.1
    ).to(device)
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        writer=writer,
        optimizer=optimizer,
        device=device,
        cpu = torch.device('cpu'),
    )
    batch_size = find_best_batch_size(trainer, dataset)
    dataset.train_loader = torch.utils.data.DataLoader(dataset.train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(initial_epoch, config.num_epochs):
        model.train()
        batch_iterator = tqdm(
            dataset.train_loader,
            desc=f"Processing epoch {epoch:02d}"
        )
        
        batch: MyBatch
        for batch in batch_iterator:
            do_one_batch(batch, batch_iterator, trainer, global_step)
            gc.collect()
    
            # if global_step % 150 == 0:
            #     model.eval()
            #     with torch.no_grad():
            #         run_validation(
            #             m=m,
            #             validation_dataset=dataset.validation_loader,
            #             max_len=config.seq_len,
            #             print_msg=lambda msg: batch_iterator.write(msg),
            #             global_step=global_step,
            #             writer=writer,
            #         )
            #    model.train()
    
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

    config = get_config()
    dataset_raw = get_dataset_raw(config)
    tokenizers = get_or_build_tokenizers(config, dataset_raw)
    dataset = tokenize_dataset(config, dataset_raw, tokenizers)
    
    train_model(config, dataset, tokenizers)

