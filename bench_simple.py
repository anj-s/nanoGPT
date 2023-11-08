"""
A much shorter version of train.py for benchmarking. 

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py config/bench_gpt2_ddp.py

Additions from anj-s: Support for other distributed APIs (+other techniques) beyond DDP.

"""
import os
from contextlib import nullcontext
import numpy as np
import time
import torch
from torch import nn
from model import GPTConfig, GPT
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from dist_initialize import distributed_init, global_barrier
from fairscale.nn.megatron.tensor_parallel.layers import RowParallelLinear

# -----------------------------------------------------------------------------
# Model definition to get started, taking a simple MLP module from the larger GPT one.
class MLP(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        # self.c_fc    = nn.Linear(d_model, 4 * d_model, bias=False)
        self.c_fc    = RowParallelLinear(d_model, 4 * d_model, bias=False)
        self.gelu    = nn.GELU()
        # self.c_proj  = nn.Linear(4 * d_model, d_model, bias=False)
        # self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        # x = self.c_proj(x)
        # x = self.dropout(x)
        return x


# -----------------------------------------------------------------------------
batch_size = 12
block_size = 1024
bias = False
real_data = False
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
profile = False # use pytorch profiler, or just simple benchmarking?
ddp = False
ddp_tp = True
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# ----------------------------------------------------------------------------------------
# Distributed settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# various inits, derived attributes, I/O setup
distributed = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?

if not distributed and ddp:
    raise RuntimeError("Initialize a distributed run.")

# ----------------------------------------------------------------------------------------
# DDP configs
if ddp:
    init_process_group(backend=backend)
    world_size = int(os.environ['WORLD_SIZE'])
    ddp_rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
elif ddp_tp:
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{local_rank}'
    torch.cuda.set_device(device)
    distributed_init(2, 1, 2)
    master_process = torch.distributed.get_rank() == 0 # this process will do logging, checkpointing etc.
    seed_offset = local_rank # each process gets a different seed
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    world_size = 1

tokens_per_iter = world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

# ----------------------------------------------------------------------------------------

# data loading init
if real_data:
    dataset = 'openwebtext'
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    def get_batch(split):
        data = train_data # note ignore split in benchmarking script
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x, y
else:
    # alternatively, if fixed data is desired to not care about data loading
    x = torch.randint(50304, (batch_size, block_size), device=device)
    y = torch.randint(50304, (batch_size, block_size), device=device)
    get_batch = lambda split: (x, y)

# model init
# gptconf = GPTConfig(
#     block_size = block_size, # how far back does the model look? i.e. context size
#     n_layer = 12, n_head = 12, n_embd = 768, # size of the model
#     dropout = 0, # for determinism
#     bias = bias,
# )
# model = GPT(gptconf)
model = MLP(8)
optimizer = model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device_type)
model.to(device)

global_barrier()

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[local_rank])
elif ddp_tp:
    pass

if compile:
    print("Compiling model...")
    model = torch.compile(model) # pytorch 2.0

if profile:
    # useful docs on pytorch profiler:
    # - tutorial https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
    # - api https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile
    wait, warmup, active = 5, 5, 5
    num_steps = wait + warmup + active
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./bench_log'),
        record_shapes=False,
        profile_memory=False,
        with_stack=False, # incurs an additional overhead, disable if not needed
        with_flops=True,
        with_modules=False, # only for torchscript models atm
    ) as prof:
    
        X, Y = get_batch('train')
        for k in range(num_steps):
            with ctx:
                logits, loss = model(X, Y)
            X, Y = get_batch('train')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lossf = loss.item()
            print(f"{k}/{num_steps} loss: {lossf:.4f}")

            prof.step() # notify the profiler at end of each step

else:

    # simple benchmarking
    torch.cuda.synchronize()
    for stage, num_steps in enumerate([10, 20]): # burnin, then benchmark
        t0 = time.time()
        X, Y = get_batch('train')
        for k in range(num_steps):
            with ctx:
                logits, loss = model(X, Y)
            X, Y = get_batch('train')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lossf = loss.item()
            print(f"{k}/{num_steps} loss: {lossf:.4f}")
        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1-t0
        mfu = model.estimate_mfu(batch_size * 1 * num_steps, dt)
        if stage == 1:
            print(f"time per iteration: {dt/num_steps*1000:.4f}ms, MFU: {mfu*100:.2f}%")
