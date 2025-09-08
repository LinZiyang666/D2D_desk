from dataclasses import dataclass
from typing import Callable, Dict, List

@dataclass(frozen=True)
class Device:
    """
    A single device in the device group.

    Attributes
    ----------
    name : str
        Unique device identifier.
    memory_budget : int
        Peak memory budget for this step (in bytes or arbitrary units).
    capacity : float
        Computing capacity v_d (higher = faster). In the paper it's defined as
        the inverse of the FP+BP latency with a micro-batch. You may precompute
        it from your profiler and fill here.
    """
    name: str
    memory_budget: int
    capacity: float

def allocate_microbatch_samples(
    group: List["Device"],
    micro_batch_size: int,
    mem_of: Callable[["Device", int], int],
    lat_of: Callable[["Device", int], float],
    block_size: int = 1,
) -> Dict[str, int]:
    def max_batch_size_under_budget(d: "Device") -> int:
        lo, hi = 0, micro_batch_size
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if mem_of(d, mid) <= d.memory_budget:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def memory_aware_balancing(group: List["Device"], beta: int, y: Dict[str, int]) -> None:
        remaining = beta
        while remaining > 0:
            headroom: Dict[str, int] = {}
            caps: Dict[str, float] = {}
            total_cap = 0.0

            for d in group:
                bs_total = max_batch_size_under_budget(d)
                avail = bs_total - y[d.name]
                if avail > 0:
                    headroom[d.name] = avail
                    v_d = 1.0 / max(1e-9, lat_of(d, micro_batch_size))
                    caps[d.name] = v_d
                    total_cap += v_d

            if not headroom:
                raise ValueError("Micro-batch cannot fit into the group's memory budgets.")

            alloc_floor: Dict[str, int] = {name: 0 for name in headroom}
            frac_part: Dict[str, float] = {name: 0.0 for name in headroom}
            assigned = 0
            if total_cap > 0.0:
                for name, v_d in caps.items():
                    share = remaining * (v_d / total_cap)
                    base = int(share)  # floor
                    give = min(base, headroom[name])
                    alloc_floor[name] = give
                    frac_part[name] = share - base
                    assigned += give

            remaining_after_floor = remaining - assigned

            if remaining_after_floor > 0:
                order = sorted(frac_part.items(), key=lambda kv: kv[1], reverse=True)
                for name, _ in order:
                    if remaining_after_floor == 0:
                        break
                    if alloc_floor[name] < headroom[name]:
                        alloc_floor[name] += 1
                        remaining_after_floor -= 1


            for name, add in alloc_floor.items():
                y[name] += add

            remaining = remaining_after_floor

           
    def straggler_workload_offloading(group: List["Device"], y: Dict[str, int]) -> None:
        def time_of(d: "Device", yd: int) -> float:
            return 0.0 if yd <= 0 else lat_of(d, yd)

        while True:
            times = [(d, time_of(d, y[d.name])) for d in group]
            times.sort(key=lambda t: t[1], reverse=True)
            slow_dev, slow_t = times[0]
            recv_candidates = [t[0] for t in reversed(times) if max_batch_size_under_budget(t[0]) - y[t[0].name] >= block_size]
            if not recv_candidates or y[slow_dev.name] < block_size:
                break
            fast_dev = recv_candidates[0]

            move = min(block_size, y[slow_dev.name], max_batch_size_under_budget(fast_dev) - y[fast_dev.name])
            if move <= 0:
                break
            y[slow_dev.name] -= move
            y[fast_dev.name] += move


            new_slow_t = max(time_of(d, y[d.name]) for d in group)
            if new_slow_t >= slow_t - 1e-12:

                y[slow_dev.name] += move
                y[fast_dev.name] -= move
                break

    if micro_batch_size <= 0:
        return {d.name: 0 for d in group}

    y: Dict[str, int] = {d.name: 0 for d in group}

    memory_aware_balancing(group, micro_batch_size, y)

    straggler_workload_offloading(group, y)

    assert sum(y.values()) == micro_batch_size, f"Allocation sum {sum(y.values())} != B {micro_batch_size}"
    return y

