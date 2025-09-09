import os
import gc
import io
import sys
import json
import time
import math
import types
import threading
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# ---- Optional psutil for process RSS ----
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

# ---- Optional torch ----
try:
    import torch  # type: ignore
except Exception:
    torch = None  # torch may not be available in the interpreter

def _now_s() -> float:
    return time.time()

def _fmt_bytes(n: int) -> str:
    if n is None:
        return "N/A"
    units = ["B","KB","MB","GB","TB"]
    i = 0
    x = float(n)
    while x >= 1024.0 and i < len(units)-1:
        x /= 1024.0
        i += 1
    return f"{x:.2f}{units[i]}"

def _get_rank() -> int:
    # Try torch.distributed first
    if torch is not None:
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return int(dist.get_rank())
        except Exception:
            pass
    # Fall back to environment heuristics
    for k in ("RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK", "SLURM_PROCID"):
        v = os.environ.get(k, None)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
    return 0

def _get_local_rank() -> int:
    # Common launchers set LOCAL_RANK
    for k in ("LOCAL_RANK", "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK"):
        v = os.environ.get(k, None)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
    return 0

def process_rss_bytes() -> Optional[int]:
    """Return the process resident memory size in bytes (None if unknown)."""
    try:
        if psutil is not None:
            return int(psutil.Process(os.getpid()).memory_info().rss)
        # Fallback for Linux without psutil
        if os.name == "posix" and os.path.exists("/proc/self/statm"):
            with open("/proc/self/statm", "r") as f:
                parts = f.read().strip().split()
                if parts:
                    pages = int(parts[1])
                    page_size = os.sysconf("SC_PAGE_SIZE")  # bytes
                    return pages * page_size
        return None
    except Exception:
        return None

def tensor_nbytes(t: Any) -> Optional[int]:
    """Best-effort number of bytes consumed by a tensor's storage (not counting grads separately)."""
    try:
        # Newer PyTorch has untyped_storage().nbytes()
        if hasattr(t, "untyped_storage"):
            s = t.untyped_storage()
            if hasattr(s, "nbytes"):
                return int(s.nbytes())
        # Fallback
        if hasattr(t, "element_size") and hasattr(t, "nelement"):
            return int(t.element_size() * t.nelement())
    except Exception:
        pass
    return None

def sizeof(obj: Any, seen: Optional[set] = None) -> int:
    """
    Recursively sum memory of tensors within Python containers.
    For tensors, count storage bytes once per unique storage pointer.
    For other Python objects, count their shallow __sizeof__.
    """
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    # Tensors (CPU or CUDA) via torch if available
    if torch is not None and isinstance(obj, torch.Tensor):
        n = tensor_nbytes(obj)
        return 0 if n is None else n

    # Storage-like? (best-effort; skip double counting)
    if torch is not None and hasattr(obj, "__class__"):
        cname = obj.__class__.__name__.lower()
        if "storage" in cname:
            # Avoid double counting storages directly
            return 0

    # Containers
    if isinstance(obj, dict):
        total = sys.getsizeof(obj)
        for k, v in obj.items():
            total += sizeof(k, seen)
            total += sizeof(v, seen)
        return total
    if isinstance(obj, (list, tuple, set)):
        total = sys.getsizeof(obj)
        for it in obj:
            total += sizeof(it, seen)
        return total

    # Other python objects
    try:
        return sys.getsizeof(obj)
    except Exception:
        return 0

def _describe_tensor(t: Any) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    try:
        d["id"] = id(t)
        d["nbytes"] = tensor_nbytes(t)
        d["shape"] = tuple(int(x) for x in getattr(t, "shape", []) or [])
        d["dtype"] = str(getattr(t, "dtype", None))
        d["device"] = str(getattr(t, "device", None))
        d["requires_grad"] = bool(getattr(t, "requires_grad", False))
        # Minimal grad_fn info
        gf = getattr(t, "grad_fn", None)
        if gf is not None:
            d["grad_fn"] = gf.__class__.__name__
        else:
            d["grad_fn"] = None
        # Base view info
        base = getattr(t, "_base", None)
        d["is_view"] = bool(base is not None)
        if base is not None:
            d["base_id"] = id(base)
    except Exception:
        pass
    return d

def list_live_tensors(max_items: int = 20, cuda_only: bool = True) -> List[Dict[str, Any]]:
    """Return metadata for up to max_items largest live tensors."""
    if torch is None:
        return []
    candidates: List[Tuple[int, Any]] = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor):
                if cuda_only and not obj.is_cuda:
                    continue
                n = tensor_nbytes(obj)
                if n:
                    candidates.append((n, obj))
        except Exception:
            # Some GC-tracked objects may be half-constructed; ignore
            continue
    candidates.sort(key=lambda x: x[0], reverse=True)
    out: List[Dict[str, Any]] = []
    for n, t in candidates[:max_items]:
        info = _describe_tensor(t)
        out.append(info)
    return out

def cuda_mem_stats() -> Dict[str, Optional[int]]:
    """Return current CUDA memory stats for the default/current device."""
    stats = {
        "cuda_allocated": None,
        "cuda_reserved": None,
        "cuda_max_allocated": None,
        "cuda_max_reserved": None,
    }
    if torch is None:
        return stats
    try:
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            stats["device_index"] = int(dev)
            stats["cuda_allocated"] = int(torch.cuda.memory_allocated(dev))
            stats["cuda_reserved"] = int(torch.cuda.memory_reserved(dev))
            stats["cuda_max_allocated"] = int(torch.cuda.max_memory_allocated(dev))
            stats["cuda_max_reserved"] = int(torch.cuda.max_memory_reserved(dev))
    except Exception:
        pass
    return stats

def summarize_container_tensors(container: Any, top_k: int = 20) -> Dict[str, Any]:
    """
    Walk a dict-like container (e.g., fwd_cache) and compute per-key tensor memory.
    Returns a dict with:
      - total_bytes
      - per_key: {key_repr: {"bytes": n, "count": m}}
      - top: list of top_k entries sorted by bytes desc
    """
    per_key: Dict[str, Dict[str, int]] = {}
    total = 0
    if isinstance(container, dict):
        items = container.items()
    else:
        # Try generic mapping interface
        try:
            items = container.items()
        except Exception:
            items = []

    for k, v in items:
        b = sizeof(v)
        total += b
        per_key[str(k)] = {"bytes": b, "count": 1}

    top = sorted(per_key.items(), key=lambda kv: kv[1]["bytes"], reverse=True)[:top_k]
    return {"total_bytes": total, "per_key": per_key, "top": top}

class MemoryMonitor:
    """
    Periodically records memory stats and (optionally) container breakdowns.
    Writes JSON lines to a log file and/or prints to stdout.
    """
    def __init__(
        self,
        log_path: Optional[str] = None,
        interval_s: float = 1.0,
        include_tensors: bool = True,
        top_tensors: int = 20,
        cuda_only_tensors: bool = True,
        print_stdout: bool = True,
        stdout_pad_lines: int = 1,   # NEW: extra blank lines after each stdout line
    ) -> None:
        self.log_path = log_path
        self.interval_s = max(0.05, float(interval_s))
        self.include_tensors = include_tensors
        self.top_tensors = int(top_tensors)
        self.cuda_only_tensors = bool(cuda_only_tensors)
        self.print_stdout = bool(print_stdout)
        self.stdout_pad_lines = max(0, int(stdout_pad_lines))

        self.rank = _get_rank()
        self.local_rank = _get_local_rank()
        self.pid = os.getpid()

        self._containers: Dict[str, Any] = {}
        self._sections: Dict[str, Dict[str, Any]] = {}

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._fh = None

    def register_container(self, name: str, container: Any) -> None:
        """Register a dict-like container to break down per key memory."""
        self._containers[name] = container

    def unregister_container(self, name: str) -> None:
        self._containers.pop(name, None)

    def _open_log(self) -> None:
        if self.log_path:
            # Create directory if needed
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            self._fh = open(self.log_path, "a", buffering=1, encoding="utf-8")  # line-buffered
        else:
            self._fh = None

    def _close_log(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except Exception:
                pass
            self._fh = None

    def section(self, name: str):
        """
        Context manager to mark a code region and record memory deltas.
        Usage:
            with monitor.section("fwd_step"):
                ... your code ...
        """
        monitor = self
        class _Section:
            def __enter__(self_inner):
                d = cuda_mem_stats()
                rss = process_rss_bytes()
                monitor._sections[name] = {
                    "start_time": _now_s(),
                    "start_cuda_allocated": d.get("cuda_allocated"),
                    "start_cuda_reserved": d.get("cuda_reserved"),
                    "start_rss": rss,
                }
                return self_inner
            def __exit__(self_inner, exc_type, exc, tb):
                d = cuda_mem_stats()
                rss = process_rss_bytes()
                sec = monitor._sections.get(name, {})
                record = {
                    "type": "section_delta",
                    "name": name,
                    "rank": monitor.rank,
                    "local_rank": monitor.local_rank,
                    "pid": monitor.pid,
                    "time": _now_s(),
                    "elapsed_s": _now_s() - sec.get("start_time", _now_s()),
                    "delta_cuda_allocated": (None if d.get("cuda_allocated") is None or sec.get("start_cuda_allocated") is None
                                             else int(d["cuda_allocated"]) - int(sec["start_cuda_allocated"])),
                    "delta_cuda_reserved": (None if d.get("cuda_reserved") is None or sec.get("start_cuda_reserved") is None
                                             else int(d["cuda_reserved"]) - int(sec["start_cuda_reserved"])),
                    "delta_rss": (None if rss is None or sec.get("start_rss") is None
                                  else int(rss) - int(sec["start_rss"])),
                }
                monitor._emit(record)
        return _Section()

    def snapshot(self) -> Dict[str, Any]:
        """Capture a one-off snapshot, returning a Python dict."""
        snap: Dict[str, Any] = {
            "type": "snapshot",
            "time": _now_s(),
            "rank": self.rank,
            "local_rank": self.local_rank,
            "pid": self.pid,
            "rss_bytes": process_rss_bytes(),
            "cuda": cuda_mem_stats(),
            "containers": {},
        }
        for name, c in self._containers.items():
            try:
                snap["containers"][name] = summarize_container_tensors(c)
            except Exception as e:
                snap["containers"][name] = {"error": str(e)}

        if self.include_tensors:
            try:
                snap["top_tensors"] = list_live_tensors(
                    max_items=self.top_tensors,
                    cuda_only=self.cuda_only_tensors,
                )
            except Exception as e:
                snap["top_tensors"] = {"error": str(e)}

        return snap

    def _emit(self, payload: Dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False)
        if self._fh is not None:
            try:
                # Keep JSONL strict in the log file (one record per line)
                self._fh.write(line + "\n")
            except Exception:
                pass
        if self.print_stdout:
            # Add extra blank lines on stdout for readability
            try:
                sys.stdout.write(line + "\n")
                if self.stdout_pad_lines > 0:
                    sys.stdout.write("\n" * self.stdout_pad_lines)
                sys.stdout.flush()
            except Exception:
                # Fallback to print if direct write fails
                print(line + ("\n" * self.stdout_pad_lines), flush=True)

    def _run(self) -> None:
        # Background polling loop
        try:
            while not self._stop.is_set():
                self._emit(self.snapshot())
                time.sleep(self.interval_s)
        finally:
            self._close_log()

    def start(self) -> None:
        """Start background polling. Idempotent."""
        if self._thread is not None:
            return
        self._open_log()
        self._stop.clear()
        t = threading.Thread(target=self._run, name=f"MemoryMonitor-r{self.rank}", daemon=True)
        self._thread = t
        t.start()

    def stop(self) -> None:
        """Stop background polling and close the log file."""
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=5.0)
        self._thread = None
        self._close_log()


# ---- Simple CLI for ad-hoc use ----
def _parse_argv(argv: List[str]) -> Dict[str, Any]:
    import argparse
    p = argparse.ArgumentParser(description="PyTorch Memory Monitor")
    p.add_argument("--log", type=str, default=None, help="Path to JSONL log file")
    p.add_argument("--interval", type=float, default=1.0, help="Polling interval seconds")
    p.add_argument("--top", type=int, default=20, help="Number of top tensors to list")
    p.add_argument("--no-tensors", action="store_true", help="Do not list individual tensors")
    p.add_argument("--include-cpu-tensors", action="store_true", help="Include CPU tensors in top list")
    p.add_argument("--pad-lines", type=int, default=1, help="Extra blank lines after each stdout line")
    p.add_argument("--duration", type=float, default=0.0, help="If >0, run for N seconds then exit")
    args = p.parse_args(argv)
    return {
        "log": args.log,
        "interval": args.interval,
        "top": args.top,
        "include_tensors": (not args.no_tensors),
        "cuda_only": (not args.include_cpu_tensors),
        "pad_lines": max(0, args.pad_lines),
        "duration": args.duration,
    }

def main_cli(argv: Optional[List[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    cfg = _parse_argv(argv)
    mon = MemoryMonitor(
        log_path=cfg["log"],
        interval_s=cfg["interval"],
        include_tensors=cfg["include_tensors"],
        top_tensors=cfg["top"],
        cuda_only_tensors=cfg["cuda_only"],
        print_stdout=True,
        stdout_pad_lines=cfg["pad_lines"],
    )
    mon.start()
    t0 = _now_s()
    try:
        if cfg["duration"] > 0:
            while _now_s() - t0 < cfg["duration"]:
                time.sleep(0.1)
        else:
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        mon.stop()

if __name__ == "__main__":
    main_cli()