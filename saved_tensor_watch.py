
"""
saved_tensor_watch.py

Pinpoint *where* the extra memory comes from by attributing autograd "saved tensors"
(the activations PyTorch keeps for backward) to modules and Python call stacks.

Key ideas:
- Use torch.autograd.graph.saved_tensors_hooks to intercept every tensor that gets
  saved for backward during forward passes.
- Without keeping references to tensors, we measure nbytes and *attribute* them to:
    * current "section" (you label it, e.g., FWD_s0_rep10)
    * current module path (via forward pre/post hooks)
    * an abbreviated Python stack (to the model code line)
- Optionally log each save/unpack event to a JSONL file; also keep in-process
  running totals so you can print "Top-N modules/sections by bytes".

Safe for CPU/gloo and CUDA; works even if you run on CPU only.
"""

import os
import sys
import time
import json
import threading
import traceback
from typing import Any, Dict, List, Optional, Tuple

# ---- Optional torch ----
try:
    import torch
    from torch import nn
    from torch.autograd.graph import saved_tensors_hooks
except Exception:
    torch = None  # type: ignore

def _now_s() -> float:
    return time.time()

def _get_rank() -> int:
    # torch.distributed if available
    if torch is not None:
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return int(dist.get_rank())
        except Exception:
            pass
    # Common envs
    for k in ("RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK", "SLURM_PROCID"):
        v = os.environ.get(k)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
    return 0

def _get_local_rank() -> int:
    for k in ("LOCAL_RANK", "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK"):
        v = os.environ.get(k)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
    return 0

def _tensor_nbytes(t: Any) -> Optional[int]:
    try:
        if hasattr(t, "untyped_storage"):
            s = t.untyped_storage()
            if hasattr(s, "nbytes"):
                return int(s.nbytes())
        if hasattr(t, "element_size") and hasattr(t, "nelement"):
            return int(t.element_size() * t.nelement())
    except Exception:
        pass
    return None

def _abbr_stack(limit: int = 8) -> List[Tuple[str, int, str]]:
    """
    Return a compact Python stack: [(filename, lineno, funcname), ...]
    Strips internal torch frames and site-packages to focus on user/model code.
    """
    frames = traceback.extract_stack()[:-2]  # drop this function + caller
    out = []
    for fr in frames:
        fn = str(fr.filename)
        if ("site-packages" in fn) or ("/torch/" in fn) or ("python" in fn and "lib" in fn and "dist-packages" in fn):
            continue
        out.append((os.path.basename(fn), int(fr.lineno), fr.name))
    return out[-limit:]  # keep the tail

class SavedTensorWatch:
    """
    Install module pre/post hooks to maintain a thread-local module stack,
    and autograd saved_tensors_hooks to measure/attribute saved tensors.

    Usage:
        watcher = SavedTensorWatch(model=stage.module,
                                   log_path="/tmp/saved_tensors_r0.jsonl",
                                   min_bytes=64<<20)
        # Around the code region of interest (e.g., stage forward for a group):
        with watcher.activate(section=f"FWD_s{stage_idx}_rep{rep_id}"):
            # ... your forward code ...

        # At any time, print top contributors:
        watcher.report_top(top_k=20)

    Notes:
        - This doesn't hold references to tensors. It only logs metadata and running totals.
        - Overhead is low if min_bytes is set to a relatively large threshold.
    """
    def __init__(
        self,
        model: Optional["nn.Module"] = None,
        log_path: Optional[str] = None,
        min_bytes: int = 1<<20,        # only record tensors >= 1 MiB
        stack_limit: int = 8,
        include_stack: bool = True,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch not available; SavedTensorWatch requires torch.")
        self.model = model
        self.log_path = log_path
        self.min_bytes = int(min_bytes)
        self.stack_limit = int(stack_limit)
        self.include_stack = bool(include_stack)

        self.rank = _get_rank()
        self.local_rank = _get_local_rank()
        self.pid = os.getpid()

        # thread-local context
        self._tls = threading.local()
        self._tls.section = None
        self._tls.module_stack = []

        # module hook handles (for detach)
        self._handles: List[Any] = []

        # in-process totals
        self.totals_by_module: Dict[str, int] = {}
        self.totals_by_section: Dict[str, int] = {}
        self.totals_by_key: Dict[str, int] = {}  # module + section
        self.totals_by_shape: Dict[str, int] = {}

        self._fh = None
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            self._fh = open(self.log_path, "a", buffering=1, encoding="utf-8")  # line-buffered

        if self.model is not None:
            self.attach_model(self.model)

    # ---------- Module stack management ----------
    def _mod_pre(self, module: "nn.Module", inputs):
        path = getattr(module, "_swt_path", None)
        if path is None:
            # Try to infer a stable path; fallback to class name
            path = getattr(module, "_get_name", lambda: module.__class__.__name__)()
        self._tls.module_stack.append(str(path))

    def _mod_post(self, module: "nn.Module", inputs, output):
        if self._tls.module_stack:
            self._tls.module_stack.pop()

    def attach_model(self, model: "nn.Module") -> None:
        """
        Recursively register forward pre/post hooks so we know "which module we are in"
        when a saved tensor is recorded.
        """
        # Build a dotted path for each submodule for better attribution
        for name, sub in model.named_modules():
            try:
                setattr(sub, "_swt_path", name if name else sub.__class__.__name__)
            except Exception:
                pass
            try:
                h1 = sub.register_forward_pre_hook(self._mod_pre, with_kwargs=False)
                h2 = sub.register_forward_hook(self._mod_post, with_kwargs=False)
                self._handles.extend([h1, h2])
            except Exception:
                continue

    def detach_model(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles = []

    # ---------- Autograd saved tensor hooks ----------
    def _pack(self, t: Any):
        try:
            n = _tensor_nbytes(t)
            if n is not None and n >= self.min_bytes:
                section = getattr(self._tls, "section", None)
                mod_path = ">".join(getattr(self._tls, "module_stack", [])) or "<no-module>"
                rec: Dict[str, Any] = {
                    "type": "saved_tensor",
                    "time": _now_s(),
                    "rank": self.rank,
                    "local_rank": self.local_rank,
                    "pid": self.pid,
                    "section": section,
                    "module": mod_path,
                    "shape": tuple(int(x) for x in getattr(t, "shape", []) or []),
                    "dtype": str(getattr(t, "dtype", None)),
                    "device": str(getattr(t, "device", None)),
                    "requires_grad": bool(getattr(t, "requires_grad", False)),
                    "nbytes": int(n),
                }
                if self.include_stack:
                    rec["stack"] = _abbr_stack(self.stack_limit)

                # update in-process totals
                self.totals_by_module[mod_path] = self.totals_by_module.get(mod_path, 0) + int(n)
                if section is not None:
                    self.totals_by_section[section] = self.totals_by_section.get(section, 0) + int(n)
                    key = f"{section}|{mod_path}"
                else:
                    key = f"<none>|{mod_path}"
                self.totals_by_key[key] = self.totals_by_key.get(key, 0) + int(n)
                shape_key = str(rec["shape"])
                self.totals_by_shape[shape_key] = self.totals_by_shape.get(shape_key, 0) + int(n)

                # optional JSONL logging
                if self._fh is not None:
                    try:
                        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    except Exception:
                        pass
        except Exception:
            pass
        return t  # IMPORTANT: we must return the original tensor unchanged

    def _unpack(self, obj):
        # We don't modify behavior; could log "used during backward" if needed.
        return obj

    # ---------- Public activation context ----------
    class _Ctx:
        def __init__(self, outer: "SavedTensorWatch", section: Optional[str] = None):
            self.outer = outer
            self.section = section
            self.ctx = None
        def __enter__(self):
            self.outer._tls.section = self.section
            self.ctx = saved_tensors_hooks(self.outer._pack, self.outer._unpack)
            return self.ctx.__enter__()
        def __exit__(self, et, ev, tb):
            try:
                if self.ctx is not None:
                    self.ctx.__exit__(et, ev, tb)
            finally:
                # clear section
                self.outer._tls.section = None
            return False

    def activate(self, section: Optional[str] = None):
        """
        Context manager: within this region, any autograd-saved tensor will be
        attributed to `section` and current module stack.
        """
        if torch is None:
            raise RuntimeError("PyTorch not available.")
        return SavedTensorWatch._Ctx(self, section)

    def report_top(self, top_k: int = 20) -> str:
        """
        Return a human-readable multi-line string of top contributors.
        """
        def _fmt(items):
            items = sorted(items.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
            lines = []
            for k, v in items:
                gb = v / (1024**3)
                lines.append(f"{gb:8.3f} GiB  |  {k}")
            return "\n".join(lines) if lines else "(empty)"
        out = []
        out.append("== Top by SECTION ==")
        out.append(_fmt(self.totals_by_section))
        out.append("\n== Top by MODULE ==")
        out.append(_fmt(self.totals_by_module))
        out.append("\n== Top by SECTION|MODULE ==")
        out.append(_fmt(self.totals_by_key))
        out.append("\n== Top by SHAPE ==")
        out.append(_fmt(self.totals_by_shape))
        s = "\n".join(out)
        print(s, flush=True)
        return s

    def close(self) -> None:
        try:
            if self._fh is not None:
                self._fh.close()
        except Exception:
            pass
        self._fh = None


# ------- Minimal demo (optional) -------
def _demo():
    if torch is None:
        print("torch not available")
        return
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(8, 8)
        def forward(self, x):
            return torch.silu(self.l(x)) * x

    m = Tiny()
    w = SavedTensorWatch(m, log_path=None, min_bytes=1, stack_limit=6)
    x = torch.randn(1024, 8, requires_grad=True)
    with w.activate(section="demo"):
        y = m(x)
    y.sum().backward()
    w.report_top()

if __name__ == "__main__":
    _demo()
