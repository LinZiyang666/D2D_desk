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
import time, math

from stage_with_mutiple_ranks import PipelineStage_with_mutiple_ranks
from schedule_runtime import PipelineScheduleRuntimeWithDirection
from pipelining_source_code.schedules import _Action, _ComputationType

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from simple_1F1B_Action import generate_1f1b_pipeline_actions
from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration

# ==== 加载 Thinker（Omni-3B，只要文本主干 + 多模态编码器） ====
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerForConditionalGeneration
from transformers import AutoProcessor

import torch, torch.nn as nn, torch.nn.functional as F

def build_causal(mask_len, device):
    return torch.triu(torch.full((mask_len, mask_len), float("-inf"), device=device), diagonal=1)\
                .unsqueeze(0).unsqueeze(0)  # [1,1,T,T]

def pack_modalities(text_embeds, audio_seq=None, vision_seq=None):

    def _ensure_3d(x):
        if x is None:
            return None
        return x if x.dim() == 3 else x.unsqueeze(0)  # [T,D] -> [1,T,D]
    seqs = [_ensure_3d(x) for x in [text_embeds, audio_seq, vision_seq] if x is not None]
    return torch.cat(seqs, dim=1) if len(seqs) > 1 else seqs[0]



class Stage0(nn.Module):
    """0a: Text embed  | 0b: Audio encoder → 2048 | 0c: Vision encoder → 2048"""
    def __init__(self, text_model, audio_enc=None, vision_enc=None):
        super().__init__()
        self.text_model = text_model
        self.embed_tokens = text_model.embed_tokens
        self.audio_enc = audio_enc
        self.vision_enc = vision_enc

        cfg = getattr(text_model, "config", None)
        self.hidden_size = getattr(cfg, "hidden_size", 2048)
        self.image_token_id = getattr(cfg, "image_token_id", None)
        self.audio_token_id = getattr(cfg, "audio_token_id", None)
        self.video_token_id = getattr(cfg, "video_token_id", None)

    @staticmethod
    def _cat_pixel_values(vision_inputs):
        """把 collate_fn 拆成的逐样本列表拼回总二维 [sum_i Ni, C_feat]；并取 grid_thw。"""
        if vision_inputs is None or not isinstance(vision_inputs, dict):
            return None, None
        if "pixel_values_list" in vision_inputs:
            pv_list = vision_inputs["pixel_values_list"]
            if pv_list is None or len(pv_list) == 0:
                return None, vision_inputs.get("grid_thw", None)
            pv = torch.cat(pv_list, dim=0)
            return pv, vision_inputs.get("grid_thw", None)
        return vision_inputs.get("pixel_values", None), vision_inputs.get("grid_thw", None)

    @staticmethod
    def _replace_feats_by_token_id(input_ids, inputs_embeds, feats, special_token_id):
        """将 inputs_embeds 中 special_token_id 的位置替换为 feats（逐位、一一对应）。"""
        if feats is None or special_token_id is None:
            return inputs_embeds

        flat_mask = (input_ids == special_token_id).reshape(-1)
        n_tokens = int(flat_mask.sum().item())
        if n_tokens == 0:
            return inputs_embeds

        if feats.size(0) != n_tokens:
            raise RuntimeError(
                f"Feature count mismatch for token_id={special_token_id}: "
                f"tokens={n_tokens} vs feats={feats.size(0)}"
            )

        emb_flat = inputs_embeds.reshape(-1, inputs_embeds.size(-1))
        feats = feats.to(device=emb_flat.device, dtype=emb_flat.dtype)
        emb_flat[flat_mask] = feats
        return emb_flat.view_as(inputs_embeds)

    def forward(self, *args, **kwargs):
        # Handle both positional and keyword arguments flexibly
        if len(args) >= 1:
            input_ids = args[0]
            attention_mask = args[1] if len(args) > 1 else kwargs.get('attention_mask', None)
            vision_inputs = args[2] if len(args) > 2 else kwargs.get('vision_inputs', None)
            audio_inputs = args[3] if len(args) > 3 else kwargs.get('audio_inputs', None)
        else:
            input_ids = kwargs['input_ids']
            attention_mask = kwargs.get('attention_mask', None)
            vision_inputs = kwargs.get('vision_inputs', None)
            audio_inputs = kwargs.get('audio_inputs', None)

        if isinstance(input_ids, (tuple, list)):
            input_ids, vision_inputs = input_ids

        B, T = input_ids.shape
        device = input_ids.device

        # 1) 文本嵌入
        hidden = self.embed_tokens(input_ids)

        # 2) 视觉编码
        if self.vision_enc is not None and vision_inputs is not None:
            pixel_values, grid_thw = self._cat_pixel_values(vision_inputs)
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)
                if hasattr(self.vision_enc, "get_dtype"):
                    pixel_values = pixel_values.type(self.vision_enc.get_dtype())
                image_embeds = self.vision_enc(pixel_values, grid_thw=grid_thw)
                hidden = self._replace_feats_by_token_id(input_ids, hidden, image_embeds, self.image_token_id)

        # 3) 音频编码
        if self.audio_enc is not None and audio_inputs is not None:
            audio_values = None
            if isinstance(audio_inputs, dict):
                audio_values = (audio_inputs.get("input_values")
                                or audio_inputs.get("audio_values")
                                or audio_inputs.get("input_features"))
            else:
                audio_values = audio_inputs
            if audio_values is not None:
                audio_values = audio_values.to(device)
                if hasattr(self.audio_enc, "get_dtype"):
                    audio_values = audio_values.type(self.audio_enc.get_dtype())
                try:
                    audio_embeds = self.audio_enc(audio_values)
                except TypeError:
                    audio_embeds = self.audio_enc(audio_values, None)
                hidden = self._replace_feats_by_token_id(input_ids, hidden, audio_embeds, self.audio_token_id)

        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.long, device=device)
        causal = build_causal(T, device=device)
        attn_4d = causal.expand(B, -1, -1, -1).clone()
        pad = (attention_mask == 0).view(B, 1, 1, T)
        attn_4d = attn_4d.masked_fill(pad, float("-inf")).contiguous()

        grid_thw = None
        if vision_inputs is not None and "grid_thw" in vision_inputs:
            grid_thw = vision_inputs["grid_thw"]
        
        if hasattr(self.text_model, "get_rope_index"):
            position_ids, _ = self.text_model.get_rope_index(
                input_ids, 
                image_grid_thw=grid_thw,
                video_grid_thw=None,
                attention_mask=attention_mask
            )
        else:
            base_pos = torch.arange(T, device=device).unsqueeze(0).repeat(B, 1)
            position_ids = torch.stack([base_pos, base_pos, base_pos], dim=0).contiguous()

        return hidden.contiguous(), attn_4d.contiguous(), position_ids.contiguous()


class Stage1(nn.Module):
    def __init__(self, text_model, L1):
        super().__init__()
        self.layers = nn.ModuleList(text_model.layers[:L1])
        self.rotary_emb = text_model.rotary_emb
        
    def forward(self, *args, **kwargs):
        # Handle flexible arguments
        if len(args) >= 3:
            hidden, attn_mask, position_ids = args[:3]
        else:
            hidden = args[0] if len(args) > 0 else kwargs['hidden']
            attn_mask = args[1] if len(args) > 1 else kwargs['attn_mask']
            position_ids = args[2] if len(args) > 2 else kwargs['position_ids']
        
        # 处理position_ids维度
        if position_ids.dim() == 2:
            position_ids = position_ids.unsqueeze(0).repeat(3, 1, 1)
        
        pos_emb = self.rotary_emb(hidden, position_ids)
        for blk in self.layers:
            hidden = blk(
                hidden_states=hidden,
                attention_mask=attn_mask,
                position_ids=position_ids,
                position_embeddings=pos_emb,
                output_attentions=False,
                use_cache=False
            )[0]
        return hidden.contiguous(), attn_mask.contiguous(), position_ids.contiguous()


class Stage2(nn.Module):
    def __init__(self, text_model, L1, L2):
        super().__init__()
        self.layers = nn.ModuleList(text_model.layers[L1:L2])
        self.rotary_emb = text_model.rotary_emb
        
    def forward(self, *args, **kwargs):
        # Handle flexible arguments
        if len(args) >= 3:
            hidden, attn_mask, position_ids = args[:3]
        else:
            hidden = args[0] if len(args) > 0 else kwargs['hidden']
            attn_mask = args[1] if len(args) > 1 else kwargs['attn_mask']
            position_ids = args[2] if len(args) > 2 else kwargs['position_ids']
        
        if position_ids.dim() == 2:
            position_ids = position_ids.unsqueeze(0).repeat(3, 1, 1)
        
        pos_emb = self.rotary_emb(hidden, position_ids)
        for blk in self.layers:
            hidden = blk(
                hidden_states=hidden,
                attention_mask=attn_mask,
                position_ids=position_ids,
                position_embeddings=pos_emb,
                output_attentions=False,
                use_cache=False
            )[0]
        return hidden.contiguous(), attn_mask.contiguous(), position_ids.contiguous()


class Stage3(nn.Module):
    def __init__(self, full_thinker, text_model, L2):
        super().__init__()
        self.layers = nn.ModuleList(text_model.layers[L2:])
        self.norm = text_model.norm
        self.lm_head = full_thinker.lm_head
        self.rotary_emb = text_model.rotary_emb
        
    def forward(self, *args, **kwargs):
        # Handle flexible arguments
        if len(args) >= 3:
            hidden, attn_mask, position_ids = args[:3]
        else:
            hidden = args[0] if len(args) > 0 else kwargs['hidden']
            attn_mask = args[1] if len(args) > 1 else kwargs['attn_mask']
            position_ids = args[2] if len(args) > 2 else kwargs['position_ids']
        
        if position_ids.dim() == 2:
            position_ids = position_ids.unsqueeze(0).repeat(3, 1, 1)
        
        pos_emb = self.rotary_emb(hidden, position_ids)
        for blk in self.layers:
            hidden = blk(
                hidden_states=hidden,
                attention_mask=attn_mask,
                position_ids=position_ids,
                position_embeddings=pos_emb,
                output_attentions=False,
                use_cache=False
            )[0]
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        return logits


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
parser.add_argument("--upstream", default=os.getenv("upstream"),
                    help='Write the upstream in mbps')
args = parser.parse_args()
def main():

    dist.init_process_group("gloo", init_method="env://")

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])

    device = torch.device("cpu")     

    MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(MODEL_ID, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    
    # 便捷访问
    text_model   = thinker.model           # 纯解码器主干（有 embed_tokens / layers / norm）
    audio_enc    = getattr(thinker, "audio_tower", None)
    vision_enc   = getattr(thinker, "visual", None)
    rotary_emb   = getattr(text_model, "rotary_emb", None)
    vocab_size   = tok.vocab_size

    # 自动切分点
    L  = len(text_model.layers)
    L1 = L // 3
    L2 = (2 * L) // 3
    
    if rank == 0:
        stage_mod = Stage0(text_model, audio_enc, vision_enc)
        stage_mod.to(device)
        stage = PipelineStage_with_mutiple_ranks(stage_mod, stage_index=0,
                            num_stages=world, device=device,
                            group=dist.group.WORLD,
                            prev_group=None, this_group=[0], next_group=[1])
        
    elif rank == 1:
        stage_mod = Stage1(text_model, L1)
        stage_mod.to(device)
        stage = PipelineStage_with_mutiple_ranks(stage_mod, stage_index=1,
                            num_stages=world, device=device,
                            group=dist.group.WORLD,
                            prev_group=[0], this_group=[1], next_group=[2])
    elif rank == 2:
        stage_mod = Stage2(text_model, L1, L2)
        stage_mod.to(device)
        stage = PipelineStage_with_mutiple_ranks(stage_mod, stage_index=2,
                            num_stages=world, device=device,
                            group=dist.group.WORLD,
                            prev_group=[1], this_group=[2], next_group=[3])
    elif rank == 3:
        stage_mod = Stage3(thinker, text_model, L2)
        stage_mod.to(device)
        stage = PipelineStage_with_mutiple_ranks(stage_mod, stage_index=3,
                            num_stages=world, device=device,
                            group=dist.group.WORLD,
                            prev_group=[2], this_group=[3], next_group=None)
    
    del thinker                        
    import gc; gc.collect()

    raw = load_dataset("jxie/flickr8k", split="train")
    
    def pick_caption(example):
        text = example.get("caption_0", None)
        if text is None:
            caps = example.get("captions") 
            text = caps[0] if isinstance(caps, list) and caps else ""
        return {"text": text}

    keep_cols = {"image", "text"}
    raw = raw.map(pick_caption, remove_columns=[c for c in raw.column_names if c not in keep_cols])

    block = 512
    def tok_fn(batch):
        out = tok(batch["text"],
                return_attention_mask=False,
                truncation=True,
                max_length=block,
                padding="max_length")

        out["labels"] = out["input_ids"].copy()
        return out

    ds = raw.map(tok_fn, batched=True)

    from PIL import Image, ImageOps

    def _round_to_multiple(x: int, m: int) -> int:
        """将数字向上取整到m的倍数"""
        return max(m, int(math.ceil(x / m) * m))

    def _pad_to_valid_size(img: Image.Image) -> Image.Image:
        """
        将图像填充到符合 Qwen2.5-Omni 视觉编码器要求的尺寸。
        
        Qwen2.5-Omni 的要求：
        - patch_size = 14
        - spatial_merge_size = 2  
        - spatial_merge_unit = 4
        - 最终 (H/14) 和 (W/14) 必须是 4 的倍数
        """
        img = ImageOps.exif_transpose(img)
        w, h = img.size

        new_w = _round_to_multiple(w, 56)
        new_h = _round_to_multiple(h, 56)

        grid_h = new_h // 14
        grid_w = new_w // 14

        while grid_h % 4 != 0:
            new_h += 56
            grid_h = new_h // 14
        
        while grid_w % 4 != 0:
            new_w += 56
            grid_w = new_w // 14
        
        new_w = max(new_w, 112)
        new_h = max(new_h, 112)
        
        pad_right = new_w - w
        pad_bottom = new_h - h
        
        if pad_right == 0 and pad_bottom == 0:
            return img
        
        # 填充图像（右边和下边）
        return ImageOps.expand(img, border=(0, 0, pad_right, pad_bottom), fill=0)

    def collate_fn(batch):

        conversations = []
        for ex in batch:
            img = ex["image"]
            txt = ex["text"] if isinstance(ex["text"], str) else ""
            conversations.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text",  "text": txt}
                    ],
                }
            ])

        pack = proc.apply_chat_template(
            conversations,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            padding="max_length", 
            max_length=512,       
            truncation=True        
        )

        input_ids      = pack["input_ids"]            # [B, T]
        attention_mask = pack["attention_mask"]       # [B, T]
        labels         = input_ids.clone()

        special_ids = set([
            tok.pad_token_id, getattr(tok, "eos_token_id", None), getattr(tok, "bos_token_id", None),
            getattr(cfg, "image_token_id", None),
            getattr(cfg, "video_token_id", None),
            getattr(cfg, "audio_token_id", None),
        ])
        special_ids = {i for i in special_ids if i is not None}
        for sid in special_ids:
            labels[labels == sid] = -100

        vision_inputs = None
        pixel_values   = pack.get("pixel_values", None)  # [sum_i N_i, C_feat]  [2940, 1176]
        image_grid_thw = torch.as_tensor(pack.get("image_grid_thw", []), dtype=torch.long)  # [B, 3]
        if pixel_values is not None and image_grid_thw.numel() > 0:
            smu = int(getattr(vision_enc, "spatial_merge_unit", 4))  # 2x2 merge -> 4
            counts = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2])
            slices, off = [], 0
            for n in counts.tolist():
                slices.append(pixel_values[off: off + n])
                off += n
            assert off == pixel_values.size(0), f"visual tokens mismatch: {off} != {pixel_values.size(0)}"

            vision_inputs = {
                "pixel_values_list": slices,  
                "grid_thw": image_grid_thw,    # [B,3]
            }

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "vision_inputs": vision_inputs,
        }






    batch_size = args.batch_size
    microbatch_num = args.microbatch_num
    block = 512
    
    if rank == 0:
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            #shuffle=True,
            drop_last=True,
            collate_fn=collate_fn
        )

    def loss_fn(output, target):
        if output is None or target is None:
            return None
        
        vocab_size = output.size(-1)
        T_logits = output.size(1)
        T_labels = target.size(1)
        T = min(T_logits, T_labels)
        
        logits = output[:, :T-1, :].reshape(-1, vocab_size)
        labels = target[:, 1:T].reshape(-1)

        valid_mask = (labels >= -100) & (labels < vocab_size)
        if not valid_mask.all():
            invalid_labels = labels[~valid_mask]
            print(f"[rank{dist.get_rank()}] WARNING: Found invalid labels: {invalid_labels[:10]}...")
        
        return F.cross_entropy(logits, labels, ignore_index=-100)

    sched = PipelineScheduleRuntimeWithDirection([stage], n_microbatches=microbatch_num,
                                                loss_fn=loss_fn, root_pass=args.sudo_pass)
    actions = generate_1f1b_pipeline_actions(num_stages=4, num_microbatches=8, upstream = args.upstream)
    sched._load_actions(actions, format="compute_comms")

    opt = optim.Adam(stage_mod.parameters(), lr=1e-4)
    prev_loss = None
    
    
    for epoch in range(1):
        if rank == 0:
            if args.train_steps is None:
                steps_tensor = torch.tensor(len(loader), device=device)
            else:
                steps_tensor = torch.tensor(args.train_steps, device=device)
            dist.broadcast(steps_tensor, src=0)
            data_iter = iter(loader)
            print(f"Total training steps: {steps_tensor.item()}")
        else:
            steps_tensor = torch.tensor(0, device=device)
            dist.broadcast(steps_tensor, src=0)

        total_steps = int(steps_tensor.item())

        if rank == 0:
            pbar = tqdm(
                total=int(total_steps),
                desc=f"Training Epoch {epoch+1}",
                unit="step",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
            start_time = time.time()

        for step in range(total_steps):
            step_start_time = time.time()
            opt.zero_grad(set_to_none=True)

            if rank == 0:
                batch = next(data_iter)
                #print(f"✅✅✅{batch}")
                inp_ids = batch["input_ids"].to(device)             # [B, block]
                attn    = batch["attention_mask"].to(device)
                vis_pack = batch["vision_inputs"]
                if vis_pack is not None:
                    if "pixel_values_list" in vis_pack:
                        vis_pack["pixel_values_list"] = [t.to(device) for t in vis_pack["pixel_values_list"]]
                    elif "pixel_values" in vis_pack:  
                        vis_pack["pixel_values"] = vis_pack["pixel_values"].to(device)
                    if torch.is_tensor(vis_pack.get("grid_thw", None)):
                        vis_pack["grid_thw"] = vis_pack["grid_thw"].to(device)


                tgt = batch["labels"].to(device)                # [B, block]

                dist.broadcast(tgt, src=0)

                sched.step(inp_ids,attention_mask=attn , vision_inputs=vis_pack, target=tgt)

            else:
                tgt = torch.zeros(batch_size, block, dtype=torch.long, device=device)
                dist.broadcast(tgt, src=0)
                sched.step(target=tgt)

            if (step + 1) % 50 == 0:
                try:
                    sched.timeline_rec.events.clear()
                except Exception:
                    pass

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
            if cur_loss is not None and rank == 0:
                print(f"[rank0] step {step+1} loss {cur_loss:.4f}")
                prev_loss = cur_loss

            dist.barrier()

        if rank == 0:
            pbar.close()
            total_time = time.time() - start_time
            print(f"\nEpoch {epoch+1} completed in {total_time:.2f}s")
            print(f"Average speed: {total_steps / total_time:.2f} steps/s")

    recv1, recv2, recv3 = [None], [None], [None]

    buf = [stage_mod.state_dict()] if rank == 1 else [None]
    dist.broadcast_object_list(buf, src=1)
    if rank == 0: part1_state = buf[0]

    buf = [stage_mod.state_dict()] if rank == 2 else [None]
    dist.broadcast_object_list(buf, src=2)
    if rank == 0: part2_state = buf[0]

    buf = [stage_mod.state_dict()] if rank == 3 else [None]
    dist.broadcast_object_list(buf, src=3)
    if rank == 0: part3_state = buf[0]

    if rank == 0:
        print("\nMerging and saving model...")
        part0_state = stage_mod.state_dict()

        merged = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(MODEL_ID, trust_remote_code=True)
        merged_state = merged.state_dict()

        for k, v in part0_state.items():
            if k.startswith("embed_tokens."):
                newk = "model.embed_tokens." + k[len("embed_tokens."):]
                merged_state[newk] = v
            elif k.startswith("rotary_emb."):
                newk = "model.rotary_emb." + k[len("rotary_emb."):]
                if newk in merged_state: 
                    merged_state[newk] = v
            elif k.startswith("audio_enc."):
                newk = "audio_tower." + k[len("audio_enc."):]
                if newk in merged_state:
                    merged_state[newk] = v
            elif k.startswith("vision_enc."):
                newk = "visual." + k[len("vision_enc."):]
                if newk in merged_state:
                    merged_state[newk] = v

        def _map_layer_key(local_key: str, global_offset: int) -> str:
            parts = local_key.split(".")
            assert parts[0] == "layers", f"unexpected key {local_key}"
            li = int(parts[1]) + global_offset
            rest = ".".join(parts[2:])
            return f"model.layers.{li}.{rest}"

        for k, v in part1_state.items():
            if k.startswith("layers."):
                newk = _map_layer_key(k, 0)
                merged_state[newk] = v

        for k, v in part2_state.items():
            if k.startswith("layers."):
                newk = _map_layer_key(k, L1)
                merged_state[newk] = v

        for k, v in part3_state.items():
            if k.startswith("layers."):
                newk = _map_layer_key(k, L2)
                merged_state[newk] = v
            elif k == "norm.weight":
                merged_state["model.norm.weight"] = v
            elif k == "norm.bias":
                if "model.norm.bias" in merged_state:
                    merged_state["model.norm.bias"] = v
            elif k == "lm_head.weight":
                merged_state["lm_head.weight"] = v
            elif k == "lm_head.bias":
                if "lm_head.bias" in merged_state:
                    merged_state["lm_head.bias"] = v

        merged.load_state_dict(merged_state, strict=False)
        merged.save_pretrained("trained_qwen_pp")
        tok.save_pretrained("trained_qwen_pp")
        print("Saved to ./trained_qwen_pp")

    dist.destroy_process_group()



if __name__ == "__main__":
    main()