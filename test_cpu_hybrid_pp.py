#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_cpu_hybrid_pp.py

This is the main experiment script for training Qwen-0.6B using a custom 
Hybrid Pipeline Parallelism framework built on top of PiPPy.

- Stage 0 (rank 0): embedding + layers 0–13
- Stage 1 (ranks 1 & 2): layers 14–27 + norm + lm_head (with DDP)

Key Features:
- CPU-only execution with 3 ranks
- Manual microbatch scheduling via custom action list
- Used for testing correctness and loss behavior of the hybrid PP runtime
"""


import argparse, os, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import torch.distributed as dist

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import time

from stage_with_mutiple_ranks import PipelineStage_with_mutiple_ranks
from schedule_runtime import PipelineScheduleRuntimeWithDirection
from pipelining_source_code.schedules import _Action, _ComputationType

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

def create_pipeline_actions():

    # [stage],[rank],[id],[action type],[microbatch],[dest_rank],[upstream],[dependency]
    # Rank 0 (Stage 0)
    rank0_actions = [
        _Action(0, 0, 0, _ComputationType.FORWARD, (0), None, None, None),
        _Action(0, 0, 1, _ComputationType.SEND_F, (0,), 1, 10000, None),
        _Action(0, 0, 2, _ComputationType.FORWARD, (1), None, None, None),
        _Action(0, 0, 3, _ComputationType.SEND_F, (1,), 2, 10000, None),
        _Action(0, 0, 4, _ComputationType.FORWARD, (2), None, None, None),
        _Action(0, 0, 5, _ComputationType.SEND_F, (2,), 1, 10000, None),
        _Action(0, 0, 6, _ComputationType.FORWARD, (3), None, None, None),
        _Action(0, 0, 7, _ComputationType.SEND_F, (3,), 2, 10000, None),
        _Action(0, 0, 8, _ComputationType.FORWARD, (4), None, None, None),
        _Action(0, 0, 9, _ComputationType.SEND_F, (4,), 1, 10000, None),

        
        _Action(0, 0, 10, _ComputationType.RECV_B, (0,), 1, None, None),
        _Action(0, 0, 11, _ComputationType.RECV_B, (1,), 2, None, None),
        _Action(0, 0, 12, _ComputationType.RECV_B, (2,), 1, None, None),
        _Action(0, 0, 13, _ComputationType.RECV_B, (3,), 2, None, None),
        _Action(0, 0, 14, _ComputationType.RECV_B, (4,), 1, None, None),
        
        _Action(0, 0, 15, _ComputationType.FULL_BACKWARD, (0), None, None, None),    
        _Action(0, 0, 16, _ComputationType.FULL_BACKWARD, (1), None, None, None),    
        _Action(0, 0, 17, _ComputationType.FULL_BACKWARD, (2), None, None, None),    
        _Action(0, 0, 18, _ComputationType.FULL_BACKWARD, (3), None, None, None),    
        _Action(0, 0, 19, _ComputationType.FULL_BACKWARD, (4), None, None, None),    
    ]
    
    # Rank 1 (Stage 1)
    rank1_actions = [
        _Action(1, 1, 0, _ComputationType.RECV_F, (0,), 0, None, None),
        _Action(1, 1, 1, _ComputationType.FORWARD, (0,), None, None, None),
        _Action(1, 1, 2, _ComputationType.RECV_F, (2,), 0, None, None),
        _Action(1, 1, 3, _ComputationType.FORWARD, (2), None, None, None),
        _Action(1, 1, 4, _ComputationType.RECV_F, (4,), 0, None, None),
        _Action(1, 1, 5, _ComputationType.FORWARD, (4), None, None, None),
        
        _Action(1, 1, 6, _ComputationType.FULL_BACKWARD, (0,), None, None, None),
        _Action(1, 1, 7, _ComputationType.SEND_B, (0,), 0, 10000, None),
        _Action(1, 1, 8, _ComputationType.FULL_BACKWARD, (2), None, None, None),
        _Action(1, 1, 9, _ComputationType.SEND_B, (2,), 0, 10000, None),
        _Action(1, 1, 10, _ComputationType.FULL_BACKWARD, (4), None, None, None),
        _Action(1, 1, 11, _ComputationType.SEND_B, (4,), 0, 10000, None),
         
        _Action(1, 1, 12, _ComputationType.ALL_REDUCE, None, None, None, None),
    ]
    
    # Rank 2 (Stage 1)
    rank2_actions = [
        _Action(1, 2, 0, _ComputationType.RECV_F, (1,), 0, None, None),
        _Action(1, 2, 1, _ComputationType.FORWARD, (1,), None, None, None),
        _Action(1, 2, 2, _ComputationType.RECV_F, (3,), 0, None, None),
        _Action(1, 2, 3, _ComputationType.FORWARD, (3,), None, None, None),
        
        _Action(1, 2, 4, _ComputationType.FULL_BACKWARD, (1), None, None, None),
        _Action(1, 2, 5, _ComputationType.SEND_B, (1,), 0, 10000, None),
        _Action(1, 2, 6, _ComputationType.FULL_BACKWARD, (3), None, None, None),
        _Action(1, 2, 7, _ComputationType.SEND_B, (3,), 0, 10000, None),

        _Action(1, 2, 8, _ComputationType.ALL_REDUCE, None, None, None, None),
    ]
    
    return {0: rank0_actions, 1: rank1_actions, 2: rank2_actions}


class Part1(nn.Module):  # rank 0
    def __init__(self, model):
        super().__init__()
        self.embed_tokens = model.model.embed_tokens
        self.layers = nn.ModuleList(model.model.layers[:14])
        self.rotary_emb = model.model.rotary_emb

    def forward(self, input_ids):
        bsz, seqlen = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1).contiguous()
        hidden = self.embed_tokens(input_ids)
        position_embeddings = self.rotary_emb(hidden, position_ids)
        attention_mask = torch.triu(
            torch.full((seqlen, seqlen), float('-inf'), device=device),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0).expand(bsz, 1, -1, -1).contiguous()

        for layer in self.layers:
            layer_outputs = layer(
                hidden_states=hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                output_attentions=False,
                use_cache=False,
            )
            hidden = layer_outputs[0]

        return hidden.contiguous(), attention_mask.contiguous()


class Part2(nn.Module):  # rank 1
    def __init__(self, model):
        super().__init__()
        self.layers = nn.ModuleList(model.model.layers[14:])
        self.norm = model.model.norm
        self.lm_head = model.lm_head
        self.rotary_emb = model.model.rotary_emb

    def forward(self, hidden, attention_mask):
        bsz, seqlen = hidden.shape[:2]
        device = hidden.device
        position_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1).contiguous()
        position_embeddings = self.rotary_emb(hidden, position_ids)

        if attention_mask.dim() == 2:
            seqlen = attention_mask.shape[-1]
            attention_mask = torch.triu(
                torch.full((seqlen, seqlen), float('-inf'), device=device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0).expand(hidden.shape[0], 1, -1, -1).contiguous()
        elif not attention_mask.is_contiguous():
            attention_mask = attention_mask.contiguous()

        for layer in self.layers:
            layer_outputs = layer(
                hidden_states=hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                output_attentions=False,
                use_cache=False,
            )
            hidden = layer_outputs[0]

        hidden = self.norm(hidden)
        return self.lm_head(hidden)

parser = argparse.ArgumentParser()
parser.add_argument("--train_steps", type=int, default=None,
                    help="The total number of steps for training. If omitted, run the entire DataLoader.")
parser.add_argument("--batch_size", type=int,
                    default=int(os.getenv("BATCH_SIZE", 16)),
                    help="The batch size of each rank (the environment variable BATCH_SIZE can be overridden)")
parser.add_argument("--microbatch_num", type=int,
                    default=int(os.getenv("MICROBATCH_NUM", 4)),
                    help="Micro-batch number (the environment variable MICROBATCH_NUM can be overridden)")
parser.add_argument("--sudo_pass", default=os.getenv("SUDO_PASS"),
                    help='Write the password of root')
args = parser.parse_args()
def main():

    dist.init_process_group("gloo", init_method="env://")

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])

    device = torch.device("cpu")     

    name = "Qwen/Qwen3-0.6B-Base"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    full = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True)

    if rank == 0:
        stage_mod = Part1(full)
        stage_mod.to(device)
        stage = PipelineStage_with_mutiple_ranks(stage_mod, stage_index=0,
                            num_stages=world, device=device,
                            group=dist.group.WORLD,
                            prev_group=None, this_group=[0], next_group=[1,2])
        
    else:
        dp_group = dist.new_group(ranks=[1, 2])
        stage_mod = Part2(full)
        stage_mod.to(device)
        
        #Using DDP as the data parallelism component of our frame
        stage_mod = DDP(
            stage_mod,
            device_ids=None,        # CPU don' use device_ids
            process_group=dp_group, # Used for local dp
            find_unused_parameters=False,
            init_sync = False,
        )
        
        # #A hook for detailed output of allreduce
        # def timing_hook(state, bucket: dist.GradBucket):
        #     """
        #     Put a dict in the state:
        #         {"pg": dp_group, "use_cuda_event": True/False}
        #     """
        #     pg = state["pg"]
        #     use_evt = bucket.buffer().is_cuda and state.get("use_cuda_event", True)
        #     if use_evt:                    # GPU, timing with cudaEvent
        #         start_evt = torch.cuda.Event(enable_timing=True)
        #         end_evt   = torch.cuda.Event(enable_timing=True)
        #         start_evt.record()
        #     else:                          # CPU or not wanting to use Event
        #         t0 = time.perf_counter()
        #      # ------- key: specify group=pg ----------
        #     work = dist.all_reduce(bucket.buffer(), group=pg, async_op=True)
        #     def _callback(fut):
        #          # timing
        #         if use_evt:
        #             end_evt.record()
        #             end_evt.synchronize()
        #             elapsed_ms = start_evt.elapsed_time(end_evt)
        #         else:
        #             elapsed_ms = (time.perf_counter() - t0) * 1e3
        #         print(f"[Rank {dist.get_rank()}] Bucket {bucket.index()} "
        #             f"all‑reduce took {elapsed_ms:.3f} ms")
        #         # Be consistent with the default behavior of DDP: take the average
        #         bucket.buffer().div_(pg.size())
        #         return bucket.buffer()
        #      # Return the Future containing _callback
        #     return work.get_future().then(_callback)
    
        # stage_mod.register_comm_hook(state={"pg": dp_group}, hook=timing_hook)
    
        stage = PipelineStage_with_mutiple_ranks(stage_mod, stage_index=1,
                                num_stages=world, device=device,
                                group=dist.group.WORLD,  # Used for world pp 
                                prev_group=[0], this_group=[1,2], next_group=None)
    
    del full                        
    import gc; gc.collect()

    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    block = 128
    def tok_fn(ex): 
        return tok(ex["text"], return_attention_mask=False)
    def grp_fn(ex):
        concat = sum(ex["input_ids"], [])
        tot = len(concat) // block * block
        ids = [concat[i:i+block] for i in range(0, tot, block)]
        return {"input_ids": ids, "labels": [x[:] for x in ids]}
    
    ds = (raw.map(tok_fn, batched=True, remove_columns=["text"])
              .map(grp_fn, batched=True))
    ds.set_format("torch", columns=["input_ids", "labels"])
    
    batch_size = args.batch_size
    microbatch_num = args.microbatch_num
    
    if rank == 0:
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    def loss_fn(output, target):
        if output is None or target is None:
            return None
        vocab_size = output.size(-1)
        output = output[:, :-1, :].reshape(-1, vocab_size)
        target = target[:, 1:].reshape(-1)
        return F.cross_entropy(output, target)

    
    sched = PipelineScheduleRuntimeWithDirection([stage], n_microbatches=microbatch_num, loss_fn=loss_fn, root_pass=args.sudo_pass)
    actions = create_pipeline_actions()
    sched._load_actions(actions, format="compute_comms")
    
    opt = optim.Adam(stage_mod.parameters(), lr=1e-4)            
    prev_loss = None
    
    
    for epoch in range(1):
        if rank == 0:
            if args.train_steps is None:
                steps_tensor = torch.tensor(len(loader), device=device)
            else:
                steps_tensor = torch.tensor(args.train_steps, device=device)
            dist.broadcast(steps_tensor, 0)
            data_iter = iter(loader)
            print(f"Total training steps: {steps_tensor.item()}")
        else:
            steps_tensor = torch.tensor(0, device=device)
            dist.broadcast(steps_tensor, 0)

        total_steps = int(steps_tensor.item())

        if rank == 0:
            pbar = tqdm(total=int(total_steps), 
                       desc=f"Training Epoch {epoch+1}",
                       unit="step",
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            start_time = time.time()
        
        for step in range(total_steps):
            step_start_time = time.time()
            opt.zero_grad(set_to_none=True)
            
            if rank == 0:
                batch = next(data_iter)
                inp = batch["input_ids"].to(device)
                tgt = batch["labels"].to(device)
                dist.broadcast(tgt, src=0)
                sched.step(inp, target=tgt)
            else:
               
                tgt = torch.empty(batch_size, block, dtype=torch.long, device=device)
                dist.broadcast(tgt, src=0)
                sched.step(target=tgt)
            
            if (step + 1) % 50 == 0:
                sched.timeline_rec.events.clear()
            opt.step()
            
            if rank == 0:
                step_time = time.time() - step_start_time
                tokens_processed = batch_size * block
                tokens_per_second = tokens_processed / step_time
                
                pbar.set_postfix({
                    'tokens/s': f'{tokens_per_second:.0f}',
                    'step_time': f'{step_time:.2f}s',
                    'lr': f'{opt.param_groups[0]["lr"]:.2e}'
                })
                pbar.update(1)
                
            cur_loss = getattr(sched, "last_step_loss", None)
            
            if cur_loss is not None:
                delta = (cur_loss - prev_loss) if prev_loss is not None else None

                print(f"[rank0] step {step+1} loss {cur_loss:.4f}"
                        + ("" if prev_loss is None else (" down" if cur_loss < prev_loss else " up" if cur_loss > prev_loss else " flat")))
                prev_loss = cur_loss
            
            
            
            dist.barrier()
        
        if rank == 0:
            pbar.close()
            total_time = time.time() - start_time
            print(f"\nEpoch {epoch+1} completed in {total_time:.2f}s")
            print(f"Average speed: {total_steps / total_time:.2f} steps/s")

    if rank == 0:
        print("\nMerging and saving model...")
        cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
        merged = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
        merged_state = merged.state_dict()
        
        part1_state = stage_mod.state_dict()
        for key in part1_state:
            if key.startswith("embed_tokens"):
                merged_state[f"model.{key}"] = part1_state[key]
            elif key.startswith("layers"):
                merged_state[f"model.{key}"] = part1_state[key]
            elif key.startswith("rotary_emb"):
                merged_state[f"model.{key}"] = part1_state[key]
        
        recv = [None]
        dist.broadcast_object_list(recv, src=1)
        part2_state = recv[0]
        
        for key in part2_state:
            if key.startswith("layers"):
                layer_idx = int(key.split(".")[1]) + 14
                new_key = f"model.layers.{layer_idx}" + key[key.find(".", 7):]
                merged_state[new_key] = part2_state[key]
            elif key == "norm.weight":
                merged_state["model.norm.weight"] = part2_state[key]
            elif key == "lm_head.weight":
                merged_state["lm_head.weight"] = part2_state[key]
                
        merged.load_state_dict(merged_state, strict=False)
        merged.save_pretrained("trained_qwen_pp")
        tok.save_pretrained("trained_qwen_pp")
        print("Saved to ./trained_qwen_pp")
    else:
        dist.broadcast_object_list([stage_mod.state_dict()], src=1)

    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()