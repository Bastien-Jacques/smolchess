# training of the GPT2 model 

import wandb #for loss logs and evaluation of the model
import argparse

import math as m
import numpy as np

from pathlib import Path 

import torch
import torch.nn.functional as F

from chess.dataset import load_tokens, iterate_batches
from chess.model import Transformer
from chess.tokenizer import PAD_ID, VOCAB_SIZE


p = argparse.ArgumentParser()

p.add_argument("--n_blocks", type=int, default=4)  
p.add_argument("--embedding_dim", type=int, default=128)
p.add_argument("--n_heads", type=int, default=4)  #on a dk = d/n_heads. On fait en sorte d'avoir dk = 64
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--schedule", type=str, default="wsd", choices=["wsd", "cosine"])
p.add_argument("--decay_start", type=int, default=4_000)
p.add_argument("--decay_steps", type=int, default=1_000)
p.add_argument("--max_lr", type=float, default=1e-3)
p.add_argument("--warmup_steps", type=int, default=200)
p.add_argument("--min_lr", type=float, default=1e-4)
p.add_argument("--weight_decay", type=float, default=0.1)
p.add_argument("--grad_clip", type=float, default=1.0)
p.add_argument("--cosine_steps", type=int, default=7000)


p.add_argument("--seed", type=int, default=0)
p.add_argument("--project", type=str, default="smolchess")

args = p.parse_args()

# Data
DATASET_CONFIG = "1M"  # "1M" to start, "default" for all 66M games
EVAL_GAMES = 5_000     # number of validation games used at each evaluation

# Model
N_BLOCKS = args.n_blocks  #amount of blocks (Attention + FeedForward) that constitutes the transformer
D = args.embedding_dim    #embedding dimension
N_HEADS = args.n_heads    #number of attention heads 



# Optimization 
BATCH_SIZE = args.batch_size        # games per batch
MAX_LR = args.max_lr                # peak learning rate 
WARMUP_STEPS = args.warmup_steps    # steps during which the LR rises from 0 to MAX_LR
MIN_LR = args.min_lr                # learning rate at the end of training
WEIGHT_DECAY = args.weight_decay    # weight decay
GRAD_CLIP = args.grad_clip          # max norm of the gradient, avoids sudden huge updates
COSINE_STEPS = args.cosine_steps    # number of steps for the cosine annealing

# Logging 
EVAL_EVERY = 250  # evaluate on the validation games every N steps
LOG_EVERY = 10    # send the training loss to wandb every N steps
SEED = args.seed
WANDB_PROJECT = args.project


def scheduler(step: int, schedule: str = "cosine", decay_start: int = 2_000, decay_steps: int = 1_000) -> float:
    """Learning rate at a given step.

    Both schedules end on the same decay line: from MAX_LR at decay_start
    down to MIN_LR at decay_start + decay_steps. The WSD enters this line at decay_start,
    the cosine enters it later, where its curve crosses the line.
    """
    # Warmup
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS

    # Main phase
    if schedule == "wsd":
        main_lr = MAX_LR
    elif schedule == "cosine":
        progress = min(1.0, (step - WARMUP_STEPS) / (COSINE_STEPS - WARMUP_STEPS))  # goes from 0 to 1
        main_lr = MIN_LR + (MAX_LR - MIN_LR) * 0.5 * (1 + m.cos(m.pi * progress))
    else:
        raise ValueError(f"Unknown schedule: {schedule!r}, expected 'wsd' or 'cosine'")

    # Shared decay line: MAX_LR before decay_start, then down to MIN_LR, then MIN_LR
    decay_progress = min(1.0, max(0.0, (step - decay_start) / decay_steps))  # 0 before, 1 after
    decay_line = MAX_LR + (MIN_LR - MAX_LR) * decay_progress

    # Follow the main phase until the decay line goes below it
    return min(main_lr, decay_line)

def compute_loss(model, batch):
    inputs = batch[:, : -1] #every token except the last one (B, T-1)
    targets = batch[: , 1:] #next token at each position     (B, T-1)
    logits = model(inputs)  #output of the model given the inputs (B, T-1, VOCAB_SIZE) 

    loss = F.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE),
        targets.reshape(-1),
        ignore_index=PAD_ID)

    real = targets != PAD_ID
    accuracy = (logits.argmax(dim=-1) == targets)[real].float().mean()
    return loss, accuracy

@torch.no_grad()
def evaluate(model, tokens, device):
    """Average loss and accuracy on games the model is not trained on."""
    model.eval() 
    losses, accuracies = [], []
    for batch in iterate_batches(tokens, BATCH_SIZE, device, shuffle=False):
        loss, accuracy = compute_loss(model, batch)
        losses.append(loss.item())
        accuracies.append(accuracy.item())
    model.train()
    return float(np.mean(losses)), float(np.mean(accuracies))

def main(schedule="wsd", decay_start=4_000, decay_steps=1_000):
    """Train one model with the given schedule, and log everything to wandb."""

    
    end_step = decay_start + decay_steps  # training stops when the final decay is over
    run_name = f"{schedule}_lr{MAX_LR:g}_trigger{decay_start}_decay{decay_steps}"

     # Stop right away on inconsistent settings, rather than after hours of training
    if schedule == "cosine" and COSINE_STEPS <= end_step:
        raise ValueError(f"cosine_steps ({COSINE_STEPS}) must be larger than the end of training ({end_step})")
    if decay_start < WARMUP_STEPS:
        raise ValueError(f"decay_start ({decay_start}) must be after the warmup ({WARMUP_STEPS} steps)")

    # Same seed for every run: same initial weights and same order of the batches,
    # so that the schedule is the only difference between two runs

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_tokens = load_tokens(DATASET_CONFIG, "train")
    val_tokens = load_tokens(DATASET_CONFIG, "validation")[:EVAL_GAMES]

    # Model and optimizer
    model = Transformer(vocab_size = VOCAB_SIZE, n_blocks = N_BLOCKS, d = D, n_heads = N_HEADS, max_len = MAX_LEN).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    n_params = sum(p.numel() for p in model.parameters())


    wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        config={
            "schedule": schedule,
            "decay_start": decay_start,
            "decay_steps": decay_steps,
            "end_step": end_step,
            "max_lr": MAX_LR,
            "min_lr": MIN_LR,
            "warmup_steps": WARMUP_STEPS,
            "cosine_steps": COSINE_STEPS,
            "batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "n_blocks": N_BLOCKS,
            "d": D,
            "n_heads": N_HEADS,
            "n_params": n_params,
            "dataset_config": DATASET_CONFIG,
            "seed": SEED,
        },
    )

    # Evaluate once before training: the starting point of the curves (about ln(1839) = 7.5)
    val_loss, val_accuracy = evaluate(model, val_tokens, device)
    wandb.log({"val/loss": val_loss, "val/accuracy": val_accuracy}, step=0)

    step = 0
    while step < end_step:  # each pass of this loop goes once through all the training games
        for batch in iterate_batches(train_tokens, BATCH_SIZE, device):
            # Learning rate of this step, given by the scheduler
            lr = scheduler(step, schedule, decay_start=decay_start, decay_steps=decay_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            # Forward pass
            loss, _ = compute_loss(model, batch)

            # Backward pass and weight update
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            step += 1

            # Logging
            if step % LOG_EVERY == 0:
                wandb.log({"train/loss": loss.item(), "train/lr": lr}, step=step)

            if step % EVAL_EVERY == 0 or step == end_step:
                val_loss, val_accuracy = evaluate(model, val_tokens, device)
                wandb.log({"val/loss": val_loss, "val/accuracy": val_accuracy}, step=step)
                print(
                    f"step {step:>5} | lr {lr:.1e} | train loss {loss.item():.3f} | "
                    f"val loss {val_loss:.3f} | val accuracy {val_accuracy:.1%} | "
                )

            if step == end_step:
                break

    # The number to compare between schedules: shown as a column in the wandb runs table
    wandb.summary["final_val_loss"] = val_loss
    wandb.summary["final_val_accuracy"] = val_accuracy

    # Save the trained weights, to play with the model later
    Path("checkpoints").mkdir(exist_ok=True)
    torch.save(model.state_dict(), Path("checkpoints") / f"{run_name}.pt")
    wandb.finish()
    return val_loss


if __name__ == "__main__":
    main(args.schedule, decay_start=args.decay_start, decay_steps=args.decay_steps)




