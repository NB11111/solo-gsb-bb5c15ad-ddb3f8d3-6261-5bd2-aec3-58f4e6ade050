"""双设备同指标并行测量比对：交接班时核对参考仪器与待检仪器的读数。

仅查询**已提交**（committed）的原始包，缓冲、重复、作废、告警与校准均不参与，
本模块不写任何数据，不改写原始包。

取数与时间：
- 按包序号（seq）升序扫描已提交包，对包内每个目标指标的测量：
  测量时间优先取测量自身的 ts，缺省取所属包的 sent_at；两者都没有则忽略；
- 同时间的测量按包序号、再按包内位置（measurements 数组下标）排序；
- 只保留测量时间落在 UTC 半开区间 [from, to) 内的候选测量；
- 单侧候选超过 200 条直接拒绝（422），不截断、不抽样。

配对规则（两侧候选分别记为 L / R，均已按时间稳定排序）：
- 仅在**单位相同**（两侧均未提供单位视为彼此相同）且
  |tL - tR| <= max_time_diff_seconds 时允许配对；
- 配对保持两侧时间顺序（非交叉），每条测量最多配对一次；
- 优化目标依次为：① 配对数最多；② 全部配对的时间差绝对值总和最小；
  ③ 仍平局时，按 (左侧排序序号, 右侧排序序号) 元组序列的字典序最小者决胜。

实现用 O(n*m) 动态规划求最优目标值，再按目标③做贪心回溯构造字典序最小的
配对序列（n, m <= 200，至多约 4 万状态）。
"""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import List, NamedTuple, Optional, Tuple

from .schemas import (
    ComparisonPair,
    ComparisonResponse,
    ComparisonSideMeasurement,
    ComparisonSummary,
)

# 单侧候选测量条数硬上限：超限明确拒绝而非截断
MAX_CANDIDATES_PER_SIDE = 200

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_US_PER_SECOND = 1_000_000


class ComparisonError(ValueError):
    """比对参数非法（时间范围倒置、设备相同、候选条数超限等）。"""


class Candidate(NamedTuple):
    """排序后的一条候选测量。"""

    time: datetime  # 已归一化为 UTC（aware）
    time_us: int  # 自 epoch 的整数微秒，时间差求和用整数避免浮点误差
    seq: int  # 所属已提交包序号
    position: int  # 包内位置，0 起（measurements 数组下标）
    value: float
    unit: Optional[str]
    time_source: str  # ts / sent_at


def _as_utc(dt: datetime) -> datetime:
    """naive 时间一律视为 UTC，保证比较行为确定。"""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_us(dt: datetime) -> int:
    """aware datetime -> 自 epoch 的整数微秒。"""
    return (dt - _EPOCH) // timedelta(microseconds=1)


def load_candidates(
    conn: sqlite3.Connection,
    device_id: str,
    metric: str,
    from_dt: datetime,
    to_dt: datetime,
) -> Tuple[List[Candidate], int]:
    """扫描某设备已提交包，取出指标在 [from, to) 内、有时间可用的候选测量。

    返回 (按 (时间, 包序号, 包内位置) 升序的候选列表, 无时间被忽略的测量数)。
    缓冲区 / 重复包不参与；本函数只读。
    """
    candidates: List[Candidate] = []
    untimed = 0
    rows = conn.execute(
        "SELECT seq, payload_json FROM committed WHERE device_id = ? ORDER BY seq",
        (device_id,),
    ).fetchall()
    for row in rows:
        payload = json.loads(row["payload_json"])
        raw_sent_at = payload.get("sent_at")
        sent_at = (
            _as_utc(datetime.fromisoformat(raw_sent_at)) if raw_sent_at else None
        )
        for position, m in enumerate(payload.get("measurements", [])):
            if m.get("metric") != metric:
                continue
            raw_ts = m.get("ts")
            if raw_ts:
                t = _as_utc(datetime.fromisoformat(raw_ts))
                source = "ts"
            elif sent_at is not None:
                t = sent_at
                source = "sent_at"
            else:
                untimed += 1
                continue  # 无时间的测量：忽略，不参与比对
            if from_dt <= t < to_dt:
                candidates.append(
                    Candidate(
                        time=t,
                        time_us=_to_us(t),
                        seq=row["seq"],
                        position=position,
                        value=float(m["value"]),
                        unit=m.get("unit"),
                        time_source=source,
                    )
                )
    candidates.sort(key=lambda c: (c.time_us, c.seq, c.position))
    return candidates, untimed


def _better(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
    """两个 (配对数, 时间差总和微秒) 取优：先比配对数（多者优），再比时间差（少者优）。"""
    if a[0] != b[0]:
        return a if a[0] > b[0] else b
    return a if a[1] <= b[1] else b


def _compatible(a: Candidate, b: Candidate, max_dt_us: int) -> bool:
    """仅当单位相同且时间差不超过上限时可配对。"""
    if a.unit != b.unit:
        return False
    return abs(a.time_us - b.time_us) <= max_dt_us


def optimal_pairs(
    left: List[Candidate], right: List[Candidate], max_dt_us: int
) -> List[Tuple[int, int]]:
    """求满足全部配对约束的最优配对，返回 (left 下标, right 下标) 升序列表。

    目标依次为配对数最多、时间差绝对值总和最小；平局取配对序号字典序最小者。
    dp[i][j] = (left[i:] 与 right[j:] 上的最优配对数, 最小时间差总和)。
    """
    n, m = len(left), len(right)
    # 边界 dp[n][*] / dp[*][m] 均为 (0, 0)：剩余侧无测量可配
    dp: List[List[Tuple[int, int]]] = [
        [(0, 0)] * (m + 1) for _ in range(n + 1)
    ]
    for i in range(n - 1, -1, -1):
        li = left[i]
        for j in range(m - 1, -1, -1):
            best = _better(dp[i + 1][j], dp[i][j + 1])  # 跳过 i 或跳过 j
            rj = right[j]
            if _compatible(li, rj, max_dt_us):
                k, cost = dp[i + 1][j + 1]
                paired = (k + 1, cost + abs(li.time_us - rj.time_us))
                best = _better(best, paired)
            dp[i][j] = best

    # 贪心回溯构造字典序最小的配对序号序列：
    # 在状态 (i, j) 上，若存在某 j' >= j 使 (i, j') 属于某个最优解，则
    # 配 i 一定得到最小的首元组（任何不配 i 的解首元组左侧序号都更大），
    # 其中取最小的 j'；否则 i 无法进入最优解，跳过 i。
    pairs: List[Tuple[int, int]] = []
    i = j = 0
    while i < n and j < m:
        li = left[i]
        chosen: Optional[int] = None
        for jp in range(j, m):
            rj = right[jp]
            if not _compatible(li, rj, max_dt_us):
                continue
            k, cost = dp[i + 1][jp + 1]
            candidate = (k + 1, cost + abs(li.time_us - rj.time_us))
            if candidate == dp[i][j]:
                chosen = jp
                break
        if chosen is None:
            i += 1  # i 在任何最优解中都不配对
        else:
            pairs.append((i, chosen))  # j..chosen-1 为右侧跳过的测量
            i += 1
            j = chosen + 1
    return pairs


def _side_out(device_id: str, c: Candidate) -> ComparisonSideMeasurement:
    return ComparisonSideMeasurement(
        device_id=device_id,
        seq=c.seq,
        measurement_index=c.position + 1,  # 对外从 1 起，符合“包内第几条”
        value=c.value,
        unit=c.unit,
        time=c.time,
        time_source=c.time_source,
    )


def compare_devices(
    conn: sqlite3.Connection,
    reference_device: str,
    target_device: str,
    metric: str,
    from_dt: datetime,
    to_dt: datetime,
    max_time_diff_seconds: float,
    value_tolerance: float,
) -> ComparisonResponse:
    """执行双设备同指标并行测量比对。两台设备的存在性由调用方预先校验。"""
    if reference_device == target_device:
        raise ComparisonError("参考设备与待检设备必须是两台不同的设备")
    from_dt = _as_utc(from_dt)
    to_dt = _as_utc(to_dt)
    if from_dt >= to_dt:
        raise ComparisonError("参数 from 必须早于 to")
    if max_time_diff_seconds < 0:
        raise ComparisonError("max_time_diff_seconds 不能为负数")
    if value_tolerance < 0:
        raise ComparisonError("value_tolerance 不能为负数")

    max_dt_us = round(max_time_diff_seconds * _US_PER_SECOND)

    left, left_untimed = load_candidates(
        conn, reference_device, metric, from_dt, to_dt
    )
    right, right_untimed = load_candidates(
        conn, target_device, metric, from_dt, to_dt
    )
    if len(left) > MAX_CANDIDATES_PER_SIDE or len(right) > MAX_CANDIDATES_PER_SIDE:
        raise ComparisonError(
            "候选测量条数超过单侧上限 "
            f"{MAX_CANDIDATES_PER_SIDE}，已拒绝（不做截断）："
            f"参考设备 {reference_device} {len(left)} 条、"
            f"待检设备 {target_device} {len(right)} 条；请缩小 UTC 时间范围后重试"
        )

    matched = optimal_pairs(left, right, max_dt_us)

    pair_outs: List[ComparisonPair] = []
    for idx, (li, rj) in enumerate(matched, start=1):
        lc, rc = left[li], right[rj]
        abs_diff = abs(lc.value - rc.value)
        pair_outs.append(
            ComparisonPair(
                index=idx,
                pair_time=lc.time,  # 配对时间以参考仪器读数时间为准
                reference=_side_out(reference_device, lc),
                target=_side_out(target_device, rc),
                abs_value_diff=abs_diff,
                time_diff_seconds=abs(lc.time_us - rc.time_us) / _US_PER_SECOND,
                over_tolerance=abs_diff > value_tolerance,
            )
        )

    summary = ComparisonSummary(
        pair_count=len(matched),
        reference_candidate_count=len(left),
        target_candidate_count=len(right),
        reference_unpaired_count=len(left) - len(matched),
        target_unpaired_count=len(right) - len(matched),
        reference_untimed_count=left_untimed,
        target_untimed_count=right_untimed,
    )
    return ComparisonResponse(
        reference_device=reference_device,
        target_device=target_device,
        metric=metric,
        **{"from": from_dt},
        to=to_dt,
        max_time_diff_seconds=max_time_diff_seconds,
        value_tolerance=value_tolerance,
        pairs=pair_outs,
        summary=summary,
    )
