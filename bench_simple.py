"""
Adapted bench.py for an example demonstrating only TP on a MLP module

To run with TP on 2 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=2 bench_tp.py

Additions from anj-s: Support for other distributed APIs (+other techniques) beyond DDP.

"""
import os
import inspect
from contextlib import nullcontext
import numpy as np
import time
import torch
from torch import nn
from model import GPTConfig, GPT
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from dist_initialize import distributed_init, global_barrier
from fairscale.nn.megatron.tensor_parallel.layers import RowParallelLinear, ColumnParallelLinear
from fairscale.nn.megatron.tensor_parallel.model_parallel_config import ModelParallelConfig

# -----------------------------------------------------------------------------
# Model definition to get started, taking a simple MLP module from the larger GPT one.
data_parallel_size = 1
tensor_parallel_size = 2
mpc = ModelParallelConfig(tensor_model_parallel_size=tensor_parallel_size)
init_fn = torch.nn.init.uniform_
class MLP(nn.Module):

    def __init__(self, d_model):
        super().__init__()
        # self.c_fc    = nn.Linear(d_model, 4 * d_model, bias=False)
        self.c_fc    = ColumnParallelLinear(d_model, 4 * d_model, config=mpc, init_method=init_fn, bias=False)
        self.gelu    = nn.GELU()
        # self.c_proj  = nn.Linear(4 * d_model, d_model, bias=False)
        self.c_proj  = RowParallelLinear(4 * d_model, d_model, config=mpc, init_method=init_fn, bias=False)
        # self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu



# -----------------------------------------------------------------------------
batch_size = 12
bias = False
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
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
    distributed_init(world_size, data_parallel_size, tensor_parallel_size)
    master_process = torch.distributed.get_rank() == 0 # this process will do logging, checkpointing etc.
    seed_offset = local_rank # each process gets a different seed
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    world_size = 1

# ----------------------------------------------------------------------------------------

# data loading init

# alternatively, if fixed data is desired to not care about data loading
x = torch.randint(50304, (batch_size, block_size), device=device)
y = torch.randint(50304, (batch_size, block_size), device=device)
get_batch = lambda split: (x, y)

# model init
model = MLP(8)
optimizer = model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device_type)
model.to(device)

global_barrier()

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[local_rank])
elif ddp_tp:
    # for now let us do vanilla TP for dp=1
    pass

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
        X = torch.randint(50304, (batch_size, 8), device=device, dtype=torch.float32)
        for k in range(num_steps):
            with ctx:
                logits = model(X)
            print(f"logits {logits}")
            break
            # loss = loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), X.view(-1), ignore_index=-1)
            # optimizer.zero_grad(set_to_none=True)
            # loss.backward()
            # optimizer.step()
            # lossf = loss.item()
            # print(f"{k}/{num_steps} loss: {lossf:.4f}")
        # torch.cuda.synchronize()
        # t1 = time.time()
        # dt = t1-t0
        # mfu = model.estimate_mfu(batch_size * 1 * num_steps, dt)
        # if stage == 1:
        #     print(f"time per iteration: {dt/num_steps*1000:.4f}ms, MFU: {mfu*100:.2f}%")
