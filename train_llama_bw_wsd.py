import os
import argparse
import time
import math
import random
import pickle
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, reduce, ReduceOp
from modelling_llama_new import build_llama_models

import datasets
import datasets.distributed
from transformers import (
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
)
from torch.utils.data import DataLoader
from itertools import cycle


base_path = '...'
dir_path = '...'
ckpt_path = '...'
print("defaulting to vocab_size to 32100")

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
device = torch.device("cuda")
# examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' # 'float32', 'bfloat16'


torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if dtype == 'float32' else torch.amp.autocast(device_type='cuda', dtype=ptdtype, cache_enabled=True)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model, eval_iters, batch_size):
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters, device=device)
    for k in range(eval_iters):
        batch_data = next(val_data)['input_ids']
        X = batch_data[:,:-1]
        Y = batch_data[:,1:]
        X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
        logits = model(X).logits.view(-1, 32100)
        loss = F.cross_entropy(logits, Y.view(-1), ignore_index=0)
        losses[k] = loss.item()
    out = losses.mean()
    reduce(out, 0, ReduceOp.AVG)
    model.train()
    return out.item()

# learning rate decay scheduler (wsd)
def get_lr(it, min_lr, max_lr, warmup_iters, max_iters, alpha=1.0, gamma=1.0):
    warmup_iters *= alpha
    middle_iters_1 = int(max_iters * 1 / 3)
    middle_iters_2 = int(max_iters * 2 / 3)
    if it <= warmup_iters:
        max_lr = max_lr * alpha * gamma
        return max_lr * it / warmup_iters
    elif it <= middle_iters_1:
        return max_lr * alpha * gamma
    elif it <= middle_iters_2:
        coeff = (max_lr * alpha * gamma - max_lr) / (middle_iters_1 - middle_iters_2)
        lr = max_lr * alpha * gamma + coeff *  (it - middle_iters_1)
        return lr
    else:
        coeff = (min_lr - max_lr) / (max_iters - middle_iters_2)
        lr = max_lr + coeff * (it - middle_iters_2)
        return lr

def train(args):

    # wandb logging
    wandb_log = args.wandb_log
    wandb_project = args.wandb_project
    wandb_run_name = args.wandb_run_name

    gradient_accumulation_steps = args.grad_micro_steps # used to simulate larger batch sizes
    batch_size = args.batch_size # if gradient_accumulation_steps > 1, this is the micro-batch size
    total_batch_size = args.total_bs
    # learning rate decay settings
    decay_lr = True # whether to decay the learning rate
    max_embed_lr =args.max_embed_lr # max learning rate
    max_head_lr =args.max_head_lr # max learning rate
    max_ln_lr =args.max_ln_lr # max learning rate
    max_qk_lr =args.max_qk_lr # max learning rate
    max_vo_lr =args.max_vo_lr # max learning rate
    max_mlp_lr =args.max_mlp_lr # max learning rate
    embed_alpha =args.embed_alpha # alpha in scheduler
    head_alpha =args.head_alpha # alpha in scheduler
    ln_alpha =args.ln_alpha # alpha in scheduler
    qk_alpha =args.qk_alpha # alpha in scheduler
    vo_alpha =args.vo_alpha # alpha in scheduler
    mlp_alpha =args.mlp_alpha # alpha in scheduler
    embed_gamma =args.embed_gamma # alpha in scheduler
    head_gamma =args.head_gamma # alpha in scheduler
    ln_gamma =args.ln_gamma # alpha in scheduler
    qk_gamma =args.qk_gamma # alpha in scheduler
    vo_gamma =args.vo_gamma # alpha in scheduler
    mlp_gamma =args.mlp_gamma # alpha in scheduler
    embed_wd =args.embed_wd # alpha in scheduler
    head_wd =args.head_wd # alpha in scheduler
    ln_wd =args.ln_wd # alpha in scheduler
    qk_wd =args.qk_wd # alpha in scheduler
    vo_wd =args.vo_wd # alpha in scheduler
    mlp_wd =args.mlp_wd # alpha in scheduler
    min_lr = args.max_lr / 10 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # min_lr = args.max_lr / 20 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    max_iters = args.max_iters # total number of training iterations
    warmup_iters = args.warmup_iters
    beta1 = args.beta1
    beta2 = args.beta2
    eps = args.eps
    use_gradpower = args.use_gradpower
    gradpower = args.gradpower
    weight_decay = args.weight_decay
    grad_clip = args.grad_clip
    log_interval = args.log_interval
    eval_interval = args.eval_interval
    save_interval = args.save_interval
    eval_iters = args.eval_iters
    resume_from_checkpoint = args.resume_from_checkpoint 
    model_name = args.model_name
    n_layer = args.n_layer
    norm_type = args.norm_type
    max_lr = args.max_lr
    

    d_input = 32100
    model = build_llama_models(model_name, d_input, block_size, device)
    print(sum(p.numel() for p in model.parameters()), ddp_rank, ddp_local_rank, world_size)

    ckpt_default_name = f'{wandb_project}_{model_name}_{max_lr}_{wandb_run_name}'

    # if master_process:
    #     for n, p in model.named_parameters():
    #         print(n)

    if resume_from_checkpoint == 'auto':
        if os.path.exists(f'{ckpt_path}/{ckpt_default_name}_ckpt.pt'):
            resume_from_checkpoint = ckpt_default_name
        else:
            resume_from_checkpoint = None
    if resume_from_checkpoint:
        ckpt = torch.load(f'{ckpt_path}/{resume_from_checkpoint}_ckpt.pt', map_location=device)
        print(f"recover from step {ckpt['iter_num']}")
        model.load_state_dict(ckpt['model'])
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    if args.faster_path:
        model = torch.compile(model)

    embed_param = [p for name, p in model.named_parameters() if 'embed' in name]
    head_param = [p for name, p in model.named_parameters() if 'lm_head' in name]
    ln_param = [p for name, p in model.named_parameters() if 'norm' in name]
    qk_param = [p for name, p in model.named_parameters() if 'q_proj' in name or 'k_proj' in name]
    vo_param = [p for name, p in model.named_parameters() if 'v_proj' in name or 'o_proj' in name]
    mlp_param = [p for name, p in model.named_parameters() if 'mlp' in name]
    extra_args = {}
    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW
        # # if args.faster_path:
        # extra_args = {'fused': True}
    elif args.optimizer == 'Lion':
        from lion import Lion 
        optimizer = Lion
    optimizer = optimizer([
        {'params': embed_param, 'lr': max_embed_lr, "name": "embed", 'weight_decay': embed_wd},
        {'params': head_param, 'lr': max_head_lr, "name": "head", 'weight_decay': head_wd},
        {'params': ln_param, 'lr': max_ln_lr, "name": "ln", 'weight_decay': ln_wd},
        {'params': qk_param, 'lr': max_qk_lr, "name": "qk", 'weight_decay': qk_wd},
        {'params': vo_param, 'lr': max_vo_lr, "name": "vo", 'weight_decay': vo_wd},
        {'params': mlp_param, 'lr': max_mlp_lr, "name": "mlp", 'weight_decay': mlp_wd},
    ], lr=max_ln_lr, betas=(beta1,beta2), weight_decay=weight_decay, **extra_args)
        
    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    if resume_from_checkpoint:
        optimizer.load_state_dict(ckpt['optimizer'])
        iter_num = ckpt['iter_num']
        del ckpt
        print("resuming dataloader....")
        for _ in range(iter_num * gradient_accumulation_steps):
            next(train_data)
        for _ in range((iter_num // eval_interval) * eval_iters):
            next(val_data)
        print("resuming done....")

    # logging
    if wandb_log and master_process:
        import wandb_wrapper as wandb
        wandb.init(project=wandb_project, name=wandb_run_name)
        config = wandb.config 
        config.total_batch_size = total_batch_size 
        config.batch_size = batch_size
        config.gradient_accumulation_steps = gradient_accumulation_steps 
        config.max_iters = max_iters
        config.warmup_iters = warmup_iters  
        config.max_embed_lr = max_embed_lr
        config.max_head_lr = max_head_lr
        config.max_ln_lr = max_ln_lr
        config.max_qk_lr = max_qk_lr
        config.max_vo_lr = max_vo_lr
        config.max_mlp_lr = max_mlp_lr
        config.embed_alpha = embed_alpha
        config.head_alpha = head_alpha
        config.ln_alpha = ln_alpha
        config.qk_alpha = qk_alpha
        config.vo_alpha = vo_alpha
        config.mlp_alpha = mlp_alpha
        config.embed_wd = embed_wd
        config.head_wd = head_wd
        config.ln_wd = ln_wd
        config.qk_wd = qk_wd
        config.vo_wd = vo_wd
        config.mlp_wd = mlp_wd
        config.beta1 = beta1
        config.beta2 = beta2
        config.eps = eps 
        config.use_gradpower = use_gradpower 
        config.gradpower = gradpower 
        config.weight_decay = weight_decay
        config.seed = args.seed
        config.log_interval = log_interval
        config.eval_interval = eval_interval
        config.save_interval = save_interval
        config.eval_iters = eval_iters
        config.grad_clip = grad_clip
        config.norm_type = norm_type

    t0 = time.time()
    losses = {"train/loss": [], "val/loss": [], 
              "train/iterval": log_interval, "val/iterval": eval_interval}

    # if master_process:
    #     for i, param_group in enumerate(optimizer.param_groups):
    #         print(param_group["name"])

    max_lrs = [max_embed_lr, max_head_lr, max_ln_lr, max_qk_lr, max_vo_lr,  max_mlp_lr]
    alphas = [embed_alpha, head_alpha, ln_alpha, qk_alpha, vo_alpha,  mlp_alpha]
    gammas = [embed_gamma, head_gamma, ln_gamma, qk_gamma, vo_gamma,  mlp_gamma]
    while True:
        # determine and set the learning rate for this iteration
        lrs = []
        for max_lr, alpha, gamma, param_group in zip(max_lrs, alphas, gammas, optimizer.param_groups):
            lr = get_lr(iter_num, min_lr, max_lr, warmup_iters, max_iters, alpha=alpha, gamma=gamma) if decay_lr else max_lr
            param_group['lr'] = lr
            lrs.append(lr)

        total_loss = 0
        # forward backward update, with optional gradient accumulation to simulate larger batch size
        for micro_step in range(gradient_accumulation_steps):
            batch_data = next(train_data)['input_ids']
            X = batch_data[:,:-1]
            Y = batch_data[:,1:]
            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            # print(X[0, 0], ddp_rank, X[1, 1])
            with ctx:
                logits = model(X).logits.view(-1, d_input)
                # logits = F.softmax(model(X).logits, dim=-1).view(-1, d_input)
            loss = F.cross_entropy(logits, Y.view(-1), ignore_index=0) # more indent
            # backward pass, with gradient scaling if training in fp16
            # scaler.scale(loss).backward()
            (loss / gradient_accumulation_steps).backward()
            total_loss += loss.detach().float() / gradient_accumulation_steps
            # step the optimizer and scaler if training in fp16
            # scaler.descent_step(optimizer, lr,max_lr) 

        if iter_num % log_interval == 0 and iter_num > 5 and master_process:
            lossf = loss.item()

        # gradient clip
        if grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # gradient power
        if use_gradpower:
            for param in model.parameters():
                if param.grad is not None:
                    g = param.grad
                    modified_g = torch.sign(g) * torch.pow(torch.abs(g) + eps, gradpower)
                    param.grad = modified_g
        
        optimizer.step()
        # scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        if iter_num % eval_interval == 0 or iter_num == max_iters:
            loss_val = estimate_loss(model, eval_iters, batch_size)
        if iter_num % log_interval == 0:
            reduce(total_loss, 0, ReduceOp.AVG)
        if iter_num % log_interval == 0 and master_process:
            lossf = total_loss.item()
            losses["train/loss"].append(lossf)

            if iter_num % eval_interval == 0 or iter_num == max_iters:
                # loss_val = estimate_loss(model, eval_iters, batch_size)
                losses["val/loss"].append(loss_val)
                print(f"iter {iter_num}: train loss {lossf:.4f}, val loss {loss_val:.4f}, time {dt*1000:.2f}ms")
                if wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "train/loss": lossf,
                        "val/loss": loss_val,
                        "embed_lr": lrs[0],
                        "head_lr": lrs[1],
                        "ln_lr": lrs[2],
                        "qk_lr": lrs[3],
                        "vo_lr": lrs[4],
                        "mlp_lr": lrs[5],
                        # "threshold": threshold
                    }, step=iter_num)
            else:
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
                if wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "train/loss": lossf,
                        "embed_lr": lrs[0],
                        "head_lr": lrs[1],
                        "ln_lr": lrs[2],
                        "qk_lr": lrs[3],
                        "vo_lr": lrs[4],
                        "mlp_lr": lrs[5],
                        # "threshold": threshold
                    }, step=iter_num)
                # else:
                    # print("no wandb log")
            
            if iter_num % save_interval == 0 and iter_num != 0:
                # checkpoint = {
                #     'model': model.module.state_dict() if ddp else model.state_dict(),
                #     'optimizer': optimizer.state_dict(),
                #     'iter_num': iter_num + 1,
                # }
                # print(f"saving checkpoint ... ", end='')
                # torch.save(checkpoint, f'{ckpt_path}/{ckpt_default_name}_tmpckpt.pt')
                # os.rename(f'{ckpt_path}/{ckpt_default_name}_tmpckpt.pt', f'{ckpt_path}/{ckpt_default_name}_ckpt.pt')
                # print(f"saving logs ... ", end='')
                with open(f"{ckpt_path}/{ckpt_default_name}_ana.p", "wb") as f:
                    pickle.dump(losses, f)
                print(f'saved')

        iter_num += 1

        t1 = time.time()
        dt = t1 - t0
        t0 = t1

        # termination conditions
        if iter_num > max_iters:
            break

    if master_process:
        with open(f"{ckpt_path}/{ckpt_default_name}_finished_ana.p", "wb") as f:
            pickle.dump(losses, f)
            print(f'saved')
        if wandb_log:
            wandb.finish()
    if ddp:
        destroy_process_group()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_log", action='store_true', help="Use Wandb Log.")
    parser.add_argument("--wandb_project", default= 'llama_c4', type=str, help="Wandb project.")
    parser.add_argument("--wandb_run_name", default='moving_4_01' , type=str, help="Wandb run name.")
    parser.add_argument("--model_name", default="0.13B", type=str)
    parser.add_argument("--n_layer", default=12, type=int, help="model depth.")
    parser.add_argument("--seed", default=41, type=int, help="Random seed.")
    parser.add_argument("--batch_size", default=15, type=int, help="Batch size.")
    parser.add_argument("--grad_micro_steps", default=10, type=int, help="Gradient accumulation steps.")
    parser.add_argument("--total_bs", default=300, type=int, help="Total batch size.")
    parser.add_argument("--log_interval", default=20, type=int, help="Log iterations.")
    parser.add_argument("--eval_interval", default=200, type=int, help="..")
    parser.add_argument("--save_interval", default=2000, type=int, help="..")
    parser.add_argument("--eval_iters", default=100, type=int, help="...")
    parser.add_argument("--max_lr", default=6e-4, type=float, help="max lr in AdamW.")
    parser.add_argument("--max_embed_lr", default=None, type=float, help="max embed lr in AdamW.")
    parser.add_argument("--max_head_lr", default=None, type=float, help="max head lr in AdamW.")
    parser.add_argument("--max_ln_lr", default=None, type=float, help="max ln lr in AdamW.")
    parser.add_argument("--max_qk_lr", default=None, type=float, help="max qk lr in AdamW.")
    parser.add_argument("--max_vo_lr", default=None, type=float, help="max vo lr in AdamW.")
    parser.add_argument("--max_mlp_lr", default=None, type=float, help="max mlp lr in AdamW.")
    parser.add_argument("--embed_alpha", default=1.0, type=float, help="embed alpha in learning rate.")
    parser.add_argument("--head_alpha", default=1.0, type=float, help="head alpha in learning rate.")
    parser.add_argument("--ln_alpha", default=1.0, type=float, help="ln alpha in learning rate.")
    parser.add_argument("--qk_alpha", default=1.0, type=float, help="qk alpha in learning rate.")
    parser.add_argument("--vo_alpha", default=1.0, type=float, help="vo alpha in learning rate.")
    parser.add_argument("--mlp_alpha", default=1.0, type=float, help="mlp alpha in learning rate.")
    parser.add_argument("--embed_gamma", default=1.0, type=float, help="embed alpha in learning rate.")
    parser.add_argument("--head_gamma", default=1.0, type=float, help="head alpha in learning rate.")
    parser.add_argument("--ln_gamma", default=1.0, type=float, help="ln alpha in learning rate.")
    parser.add_argument("--qk_gamma", default=1.0, type=float, help="qk alpha in learning rate.")
    parser.add_argument("--vo_gamma", default=1.0, type=float, help="vo alpha in learning rate.")
    parser.add_argument("--mlp_gamma", default=1.0, type=float, help="mlp alpha in learning rate.")
    parser.add_argument("--embed_wd", default=None, type=float, help="embed wd in learning rate.")
    parser.add_argument("--head_wd", default=None, type=float, help="head wd in learning rate.")
    parser.add_argument("--ln_wd", default=None, type=float, help="ln wd in learning rate.")
    parser.add_argument("--qk_wd", default=None, type=float, help="qk wd in learning rate.")
    parser.add_argument("--vo_wd", default=None, type=float, help="vo wd in learning rate.")
    parser.add_argument("--mlp_wd", default=None, type=float, help="mlp wd in learning rate.")
    parser.add_argument("--softcapping", default=0., type=float)
    parser.add_argument("--norm_type", default="pre_norm", choices=["pre_norm", "post_norm", "npost_norm", "rotate_norm", "nrotate_norm"])
    parser.add_argument("--max_iters", default=None, type=int, help="max iterations.")
    parser.add_argument("--warmup_iters", default=1000, type=int, help="warmup iterations.")
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--dataset", default='c4', type=str, choices=['c4'])
    parser.add_argument("--faster_path", action="store_true")
    parser.add_argument("--switch_iters", default=None, type=int, help="warning: only kept for compatibility")
    parser.add_argument("--beta1", default=None, type=float, help="beta1 in AdamW.")
    parser.add_argument("--beta2", default=None, type=float, help="beta2 in AdamW.")
    parser.add_argument("--eps", default=1e-8, type=float, help="epsilon in AdamW.")
    parser.add_argument("--use_gradpower", action="store_true")
    parser.add_argument("--gradpower", default=1.0, type=float, help="grad power in AdamWpower.")
    parser.add_argument("--workspace", default='llm', choices=['llm', 'moe'])
    parser.add_argument("--weight_decay", default=0.1, type=float, help="weight decay in AdamW.")
    parser.add_argument("--grad_clip", default=1.0, type=float, help="grad clip in AdamW.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--gpu_count", type=int, default=1, help="")
    parser.add_argument('--dist_url', type=str, default="")
    args = parser.parse_args()
    args.max_embed_lr = args.max_embed_lr or args.max_lr
    args.max_head_lr = args.max_head_lr or args.max_lr
    args.max_ln_lr = args.max_ln_lr or args.max_lr
    args.max_qk_lr = args.max_qk_lr or args.max_lr
    args.max_vo_lr = args.max_vo_lr or args.max_lr
    args.max_mlp_lr = args.max_mlp_lr or args.max_lr
    args.embed_wd = args.embed_wd if args.embed_wd is not None else args.weight_decay
    args.head_wd = args.head_wd if args.head_wd is not None else args.weight_decay
    args.ln_wd = args.ln_wd if args.ln_wd is not None else args.weight_decay
    args.qk_wd = args.qk_wd if args.qk_wd is not None else args.weight_decay
    args.vo_wd = args.vo_wd if args.vo_wd is not None else args.weight_decay
    args.mlp_wd = args.mlp_wd or args.weight_decay
    if args.switch_iters is not None:
        import warnings
        warnings.warn("switch_iters is only kept for compatibility reason!")

    if args.optimizer == 'AdamW':
        args.beta1 = args.beta1 or 0.9
        args.beta2 = args.beta1 or 0.95
        args.weight_decay = 0.1 if args.weight_decay is None else args.weight_decay 
    elif args.optimizer == 'Lion':
        args.beta1 = args.beta1 or 0.95
        args.beta2 = args.beta1 or 0.98
        args.weight_decay = 1. if args.weight_decay is None else args.weight_decay 
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
        
    global ddp, ddp_rank, ddp_local_rank, master_process, world_size
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?

    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = args.local_rank if args.local_rank != -1 else int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    else:
        # if not ddp, we are running on a single gpu, and one process
        ddp_rank = 0                             #ddp_rank is used in get_batch function so this has to be here also when running locally
        master_process = True
        world_size = 1

    global block_size, train_data, val_data
    block_size = 256
    args.max_iters = args.max_iters or 30000 # 50000 -> 30000 
    args.warmup_iters = args.warmup_iters or 600 # 10000 -> 600

    dataset = datasets.load_dataset(f"{dir_path}/c4", streaming=True)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"{dir_path}/t5-tokenizer/tokenizer.json")
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1

    def tokenize_fun(data):
        output = tokenizer(data["text"], truncation=True, max_length=block_size+1, padding=False)
        return output

    tokenized_data = dataset.map(tokenize_fun, batched=True, remove_columns=["text", "url", "timestamp"])
    tokenized_train_data = datasets.distributed.split_dataset_by_node(tokenized_data['train'], rank=ddp_rank, world_size=world_size)
    tokenized_val_data = datasets.distributed.split_dataset_by_node(tokenized_data['validation'], rank=ddp_rank, world_size=world_size)
    collate_fn = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    train_dataloader = DataLoader(tokenized_train_data, batch_size=args.batch_size, collate_fn=collate_fn, pin_memory=True, pin_memory_device=device)
    val_dataloader = DataLoader(tokenized_val_data, batch_size=args.batch_size, collate_fn=collate_fn, pin_memory=True, pin_memory_device=device)
    # num_workers=4
    train_data = iter(train_dataloader)
    val_data = cycle(val_dataloader)

    setup_seed(args.seed)
    train(args)

