##############################################
# 5-stage Pipeline Parts for Qwen-0.6B
# ───────────────────────────────────────────
# Stage 0 : embed + layers 0-5
# Stage 1 : layers 6-11
# Stage 2 : layers 12-17
# Stage 3 : layers 18-22
# Stage 4 : layers 23-27 + norm + lm_head
##############################################

import os, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import torch.distributed as dist

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import time

from pipelining_source_code import PipelineStage
from pipelining_source_code.schedules import ScheduleGPipe

class Part0(nn.Module):                          # rank 0
    def __init__(self, model):
        super().__init__()
        self.embed_tokens = model.model.embed_tokens
        self.layers = nn.ModuleList(model.model.layers[:6])     # 0-5
        self.rotary_emb = model.model.rotary_emb                # 共享旋转位置

    def forward(self, input_ids):
        bsz, seqlen = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1)
        hidden = self.embed_tokens(input_ids)
        pos_emb = self.rotary_emb(hidden, position_ids)

        attn_mask = torch.triu(torch.full((seqlen, seqlen), float('-inf'), device=device), 1)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(bsz, 1, -1, -1).contiguous()

        for layer in self.layers:
            hidden = layer(hidden_states=hidden,
                           attention_mask=attn_mask,
                           position_ids=position_ids,
                           position_embeddings=pos_emb,
                           output_attentions=False,
                           use_cache=False)[0]

        return hidden.contiguous(), attn_mask.contiguous()


class _MiddlePart(nn.Module):
    """公共基类：仅负责若干 transformer layer。"""
    def __init__(self, model, start, end):
        super().__init__()
        self.layers = nn.ModuleList(model.model.layers[start:end])
        self.rotary_emb = model.model.rotary_emb

    def forward(self, hidden, attn_mask):
        bsz, seqlen = hidden.shape[:2]
        device = hidden.device
        position_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1)
        pos_emb = self.rotary_emb(hidden, position_ids)

        if attn_mask.dim() == 2:                       # 兼容单矩阵传递
            attn_mask = torch.triu(
                torch.full((seqlen, seqlen), float('-inf'), device=device), 1
            ).unsqueeze(0).unsqueeze(0).expand(bsz, 1, -1, -1).contiguous()
        elif not attn_mask.is_contiguous():
            attn_mask = attn_mask.contiguous()

        for layer in self.layers:
            hidden = layer(hidden_states=hidden,
                           attention_mask=attn_mask,
                           position_ids=position_ids,
                           position_embeddings=pos_emb,
                           output_attentions=False,
                           use_cache=False)[0]
        return hidden.contiguous(), attn_mask           # 继续向后传递


class Part1(_MiddlePart):                              # rank 1：6-11
    def __init__(self, model):
        super().__init__(model, 6, 12)


class Part2(_MiddlePart):                              # rank 2：12-17
    def __init__(self, model):
        super().__init__(model, 12, 18)


class Part3(_MiddlePart):                              # rank 3：18-22
    def __init__(self, model):
        super().__init__(model, 18, 23)


class Part4(nn.Module):                                # rank 4：23-27 + norm + lm_head
    def __init__(self, model):
        super().__init__()
        self.layers = nn.ModuleList(model.model.layers[23:])
        self.norm = model.model.norm
        self.lm_head = model.lm_head
        self.rotary_emb = model.model.rotary_emb

    def forward(self, hidden, attn_mask):
        bsz, seqlen = hidden.shape[:2]
        device = hidden.device
        position_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, -1)
        pos_emb = self.rotary_emb(hidden, position_ids)

        if attn_mask.dim() == 2:
            attn_mask = torch.triu(
                torch.full((seqlen, seqlen), float('-inf'), device=device), 1
            ).unsqueeze(0).unsqueeze(0).expand(bsz, 1, -1, -1).contiguous()
        elif not attn_mask.is_contiguous():
            attn_mask = attn_mask.contiguous()

        for layer in self.layers:
            hidden = layer(hidden_states=hidden,
                           attention_mask=attn_mask,
                           position_ids=position_ids,
                           position_embeddings=pos_emb,
                           output_attentions=False,
                           use_cache=False)[0]

        hidden = self.norm(hidden)
        return self.lm_head(hidden)                     # logits


def main():
    dist.init_process_group("gloo", init_method="env://")

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])

    device = torch.device("cpu")     

    name = "Qwen/Qwen3-0.6B-Base"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    full = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True)

    world = 5
    stage_mod = None
    if rank == 0:
        stage_mod = Part0(full)
    elif rank == 1:
        stage_mod = Part1(full)
    elif rank == 2:
        stage_mod = Part2(full)
    elif rank == 3:
        stage_mod = Part3(full)
    elif rank == 4:
        stage_mod = Part4(full)
    stage_mod.to(device)

    stage = PipelineStage(stage_mod, stage_index=rank,
                          num_stages=world, device=device,
                          group=dist.group.WORLD)
    
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    block = 128
    batch_size = 5  # Define batch size consistently
    
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
    
    if rank == 0:
        loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    def loss_fn(output, target):
        if output is None or target is None:
            return None
        
        batch_size, seq_len, vocab_size = output.shape
        output = output[:, :-1, :].reshape(-1, vocab_size)
        target = target[:, 1:].reshape(-1)
        
        return F.cross_entropy(output, target)

    # Use ScheduleGPipe with the correct number of microbatches
    # Number of microbatches should be <= batch_size
    n_microbatches = min(5, batch_size)
    sched = ScheduleGPipe(stage, n_microbatches=n_microbatches, loss_fn=loss_fn)
    opt = optim.Adam(stage_mod.parameters(), lr=1e-4)

    for epoch in range(1):
        if rank == 0:
            steps = torch.tensor(len(loader), device=device)
            dist.broadcast(steps, 0)
            data_iter = iter(loader)
            print(f"Total training steps: {steps.item()}")
        else:
            steps = torch.tensor(0, device=device)
            dist.broadcast(steps, 0)

        if rank == 0:
            pbar = tqdm(total=int(steps.item()), 
                       desc=f"Training Epoch {epoch+1}",
                       unit="step",
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            start_time = time.time()
        
        for step in range(int(steps.item())):
            step_start_time = time.time()
            opt.zero_grad(set_to_none=True)
            
            if rank == 0:
                batch = next(data_iter)
                inp = batch["input_ids"].to(device)
                tgt = batch["labels"].to(device)
                
                # Broadcast the target shape info
                tgt_shape = torch.tensor([tgt.shape[0], tgt.shape[1]], device=device)
                dist.broadcast(tgt_shape, src=0)
                
                # Broadcast the actual target tensor
                dist.broadcast(tgt, src=0)
                
                # Step through the pipeline
                sched.step(inp, target=tgt)
            else:
                # Receive target shape
                tgt_shape = torch.empty(2, dtype=torch.long, device=device)
                dist.broadcast(tgt_shape, src=0)
                
                # Create empty target tensor with correct shape
                tgt = torch.empty(tgt_shape[0].item(), tgt_shape[1].item(), dtype=torch.long, device=device)
                dist.broadcast(tgt, src=0)
                
                # Step through the pipeline
                sched.step(target=tgt)
            
            opt.step()
            if rank == 0:
                tokens_processed = batch_size * block
                step_time = time.time() - step_start_time
                tokens_per_second = tokens_processed / step_time
                print("tokens:",tokens_per_second)
                pbar.set_postfix({
                    'tokens/s': f'{tokens_per_second:.0f}',
                    'step_time': f'{step_time:.2f}s',
                    'lr': f'{opt.param_groups[0]["lr"]:.2e}'
                })
                pbar.update(1)
            
            dist.barrier()
        
        if rank == 0:
            pbar.close()
            total_time = time.time() - start_time
            print(f"\nEpoch {epoch+1} completed in {total_time:.2f}s")
            print(f"Average speed: {steps.item() / total_time:.2f} steps/s")

    if rank == 0:
        print("\nMerging and saving model...")
        cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
        merged = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
        merged_state = merged.state_dict()

        # Rank 0 loads and saves its own part
        part0_state = stage_mod.state_dict()
        for key in part0_state:
            if key.startswith("embed_tokens"):
                merged_state[f"model.{key}"] = part0_state[key]
            elif key.startswith("layers"):
                merged_state[f"model.{key}"] = part0_state[key]
            elif key.startswith("rotary_emb"):
                merged_state[f"model.{key}"] = part0_state[key]

        # Receive and merge state_dicts from other ranks
        for i in range(1, 5):
            recv = [None]
            dist.broadcast_object_list(recv, src=i)
            part_state = recv[0]

            # Merge each rank's state_dict
            if i < 4:  # Parts 1-3 only have layers
                for key in part_state:
                    if key.startswith("layers"):
                        # Calculate the correct layer index
                        layer_local_idx = int(key.split(".")[1])
                        layer_global_idx = i * 6 + layer_local_idx
                        new_key = f"model.layers.{layer_global_idx}" + key[key.find(".", 7):]
                        merged_state[new_key] = part_state[key]
            else:  # Part 4 has layers + norm + lm_head
                for key in part_state:
                    if key.startswith("layers"):
                        layer_local_idx = int(key.split(".")[1])
                        layer_global_idx = 23 + layer_local_idx  # Part 4 starts at layer 23
                        new_key = f"model.layers.{layer_global_idx}" + key[key.find(".", 7):]
                        merged_state[new_key] = part_state[key]
                    elif key == "norm.weight":
                        merged_state["model.norm.weight"] = part_state[key]
                    elif key == "lm_head.weight":
                        merged_state["lm_head.weight"] = part_state[key]

        # Load the merged state_dict
        merged.load_state_dict(merged_state, strict=False)
        
        # Save the merged model
        merged.save_pretrained("trained_qwen_pp")
        tok.save_pretrained("trained_qwen_pp")
        print("Saved to ./trained_qwen_pp")

    else:
        # Broadcast own state_dict to Rank 0
        dist.broadcast_object_list([stage_mod.state_dict()], src=rank)

    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()