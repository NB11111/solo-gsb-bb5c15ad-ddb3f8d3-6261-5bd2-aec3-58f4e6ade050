"""GET /api/v1/metrics/compare 双设备同指标并行测量配对比对的测试。"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


# 所有用例共用同一个临时库，用例间用唯一设备命名空间隔离，避免读数跨用例累积
_NS = {"ref": "ref", "tgt": "tgt"}


@pytest.fixture(autouse=True)
def _unique_namespace(request):
    suffix = request.node.name.replace("_", "-")
    _NS["ref"] = f"ref-{suffix}"
    _NS["tgt"] = f"tgt-{suffix}"
    yield


def post_pkg(client, device, seq, measurements, sent_at=None, key=None):
    body = {"device_id": device, "seq": seq,
            "idempotency_key": key or f"{device}-{seq}",
            "measurements": measurements}
    if sent_at is not None:
        body["sent_at"] = sent_at
    r = client.post("/api/v1/packages", json=body)
    assert r.status_code == 200, r.text
    return r


def m(value, ts=None, unit=None, metric="temperature"):
    d = {"metric": metric, "value": value}
    if ts is not None:
        d["ts"] = ts
    if unit is not None:
        d["unit"] = unit
    return d


def compare(client, ref=None, tgt=None, **overrides):
    params = {
        "reference_device": ref or _NS["ref"],
        "target_device": tgt or _NS["tgt"],
        "metric": "temperature",
        "from": "2026-09-22T08:00:00Z",
        "to": "2026-09-22T09:00:00Z",
        "max_time_diff_seconds": 5,
        "value_tolerance": 0.5,
    }
    params.update(overrides)
    return client.get("/api/v1/metrics/compare", params=params)


def test_basic_pairing_uses_ts_and_reports_fields(client):
    post_pkg(client, _NS["ref"], 1, [
        m(20.0, "2026-09-22T08:00:00Z"),
        m(21.0, "2026-09-22T08:00:10Z"),
    ], sent_at="2026-09-22T09:30:00Z")  # sent_at 与 ts 不同，应优先 ts
    post_pkg(client, _NS["tgt"], 1, [
        m(20.2, "2026-09-22T08:00:00.100000Z"),
        m(21.6, "2026-09-22T08:00:10Z"),  # 差值 0.6 > 0.5 → 超差
    ], sent_at="2026-09-22T09:30:00Z")

    r = compare(client)
    assert r.status_code == 200, r.text
    body = r.json()
    pairs = body["pairs"]
    assert len(pairs) == 2
    p0, p1 = pairs
    assert p0["index"] == 1 and p1["index"] == 2
    # 配对时间取参考侧时间
    assert p0["pair_time"] == "2026-09-22T08:00:00Z"
    assert p0["reference"]["seq"] == 1
    assert p0["reference"]["measurement_index"] == 1
    assert p0["reference"]["time_source"] == "ts"
    assert p0["target"]["measurement_index"] == 1
    assert p0["time_diff_seconds"] == pytest.approx(0.1)
    assert p0["abs_value_diff"] == pytest.approx(0.2)
    assert p0["over_tolerance"] is False
    assert p1["abs_value_diff"] == pytest.approx(0.6)
    assert p1["over_tolerance"] is True
    s = body["summary"]
    assert s == {
        "pair_count": 2,
        "reference_candidate_count": 2,
        "target_candidate_count": 2,
        "reference_unpaired_count": 0,
        "target_unpaired_count": 0,
        "reference_untimed_count": 0,
        "target_untimed_count": 0,
    }


def test_falls_back_to_package_sent_at_and_counts_untimed(client):
    # 包 1：无测量 ts，有 sent_at → 用 sent_at
    post_pkg(client, _NS["ref"], 1, [m(20.0)], sent_at="2026-09-22T08:00:00Z")
    # 包 2：无 ts 无 sent_at → 忽略
    post_pkg(client, _NS["ref"], 2, [m(21.0)])
    post_pkg(client, _NS["tgt"], 1, [m(20.1)], sent_at="2026-09-22T08:00:02Z")

    r = compare(client)
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["pair_count"] == 1
    assert body["summary"]["reference_candidate_count"] == 1
    assert body["summary"]["reference_untimed_count"] == 1
    assert body["pairs"][0]["reference"]["time_source"] == "sent_at"


def test_unit_mismatch_blocks_pair_and_null_units_pair(client):
    post_pkg(client, _NS["ref"], 1, [m(20.0, "2026-09-22T08:00:00Z", unit="C")])
    post_pkg(client, _NS["tgt"], 1, [
        m(68.0, "2026-09-22T08:00:00Z", unit="F"),   # 单位不同，不配对
    ])
    r = compare(client)
    assert r.status_code == 200
    assert r.json()["summary"]["pair_count"] == 0

    # 有单位 vs 无单位也不配对
    post_pkg(client, _NS["ref"], 2, [m(20.0, "2026-09-22T08:05:00Z", unit="C")])
    post_pkg(client, _NS["tgt"], 2, [m(20.0, "2026-09-22T08:05:00Z")])
    r = compare(client)
    assert r.json()["summary"]["pair_count"] == 0

    # 两侧都无单位：视为单位相同，可配对
    post_pkg(client, _NS["ref"], 3, [m(20.0, "2026-09-22T08:10:00Z")])
    post_pkg(client, _NS["tgt"], 3, [m(20.0, "2026-09-22T08:10:00Z")])
    r = compare(client)
    s = r.json()["summary"]
    assert s["pair_count"] == 1
    assert s["reference_unpaired_count"] == 2
    assert s["target_unpaired_count"] == 2


def test_time_window_is_half_open(client):
    post_pkg(client, _NS["ref"], 1, [
        m(1.0, "2026-09-22T07:59:59Z"),   # 早于 from，排除
        m(2.0, "2026-09-22T08:00:00Z"),   # 恰好 from，包含
        m(3.0, "2026-09-22T08:59:59Z"),
        m(4.0, "2026-09-22T09:00:00Z"),   # 恰好 to，排除
    ])
    post_pkg(client, _NS["tgt"], 1, [
        m(2.0, "2026-09-22T08:00:00Z"),
        m(3.0, "2026-09-22T08:59:59Z"),
    ])
    body = compare(client).json()
    assert body["summary"]["reference_candidate_count"] == 2
    assert body["summary"]["target_candidate_count"] == 2
    assert body["summary"]["pair_count"] == 2


def test_same_time_sorted_by_seq_then_position(client):
    # 同时间的两个包：seq 小的先；包内位置按数组顺序
    post_pkg(client, _NS["ref"], 1, [
        m(10.0, "2026-09-22T08:00:00Z"),
        m(11.0, "2026-09-22T08:00:00Z"),
    ], sent_at="2026-09-22T08:00:00Z", key="r1")
    post_pkg(client, _NS["ref"], 2, [m(99.0, "2026-09-22T08:00:00Z")],
             sent_at="2026-09-22T08:00:00Z", key="r2")
    # tgt 同时间 3 条，按 (seq, 位置) 对齐
    post_pkg(client, _NS["tgt"], 1, [
        m(10.5, "2026-09-22T08:00:00Z"),
        m(11.5, "2026-09-22T08:00:00Z"),
        m(99.5, "2026-09-22T08:00:00Z"),
    ], sent_at="2026-09-22T08:00:00Z")
    body = compare(client).json()
    assert [p["reference"]["value"] for p in body["pairs"]] == [10.0, 11.0, 99.0]
    assert [p["target"]["value"] for p in body["pairs"]] == [10.5, 11.5, 99.5]
    # 第三条参考来自 seq=2 的包
    assert body["pairs"][2]["reference"]["seq"] == 2


def test_max_pair_count_takes_priority(client):
    # L=[0,6], R=[5,10]，d=5：
    # 贪最近会配 (6,5) 只得 1 对；最优 2 对：(0,5)+(6,10)，时间差 5+4
    post_pkg(client, _NS["ref"], 1, [
        m(1.0, "2026-09-22T08:00:00Z"),
        m(1.0, "2026-09-22T08:00:06Z"),
    ])
    post_pkg(client, _NS["tgt"], 1, [
        m(1.0, "2026-09-22T08:00:05Z"),
        m(1.0, "2026-09-22T08:00:10Z"),
    ])
    body = compare(client).json()
    assert body["summary"]["pair_count"] == 2
    assert [p["reference"]["time"] for p in body["pairs"]] == [
        "2026-09-22T08:00:00Z",
        "2026-09-22T08:00:06Z",
    ]
    assert [p["time_diff_seconds"] for p in body["pairs"]] == [5.0, 4.0]


def test_min_total_time_diff_is_second_objective(client):
    # L=[0,5,10], R=[3,8]，d=5：
    # (l0,r3)(l5,r8)=6  (l0,r3)(l10,r8)=5  (l5,r3)(l10,r8)=4 ← 最优
    post_pkg(client, _NS["ref"], 1, [
        m(1.0, f"2026-09-22T08:00:{s:02d}Z") for s in (0, 5, 10)
    ])
    post_pkg(client, _NS["tgt"], 1, [
        m(1.0, f"2026-09-22T08:00:{s:02d}Z") for s in (3, 8)
    ])
    body = compare(client).json()
    assert body["summary"]["pair_count"] == 2
    assert [p["reference"]["time"] for p in body["pairs"]] == [
        "2026-09-22T08:00:05Z",
        "2026-09-22T08:00:10Z",
    ]
    assert body["summary"]["reference_unpaired_count"] == 1  # l0 未配对


def test_tie_resolved_by_pair_index_lexicographic(client):
    # L=[0,6,12], R=[3,9]，d=12：三种 2 对方案时间差总和都为 6，
    # 配对序号字典序最小者 ((0,0),(1,1)) 胜 → l12 未配对
    post_pkg(client, _NS["ref"], 1, [
        m(1.0, f"2026-09-22T08:00:{s:02d}Z") for s in (0, 6, 12)
    ])
    post_pkg(client, _NS["tgt"], 1, [
        m(1.0, f"2026-09-22T08:00:{s:02d}Z") for s in (3, 9)
    ])
    body = compare(client, max_time_diff_seconds=12).json()
    assert body["summary"]["pair_count"] == 2
    assert [p["reference"]["time"] for p in body["pairs"]] == [
        "2026-09-22T08:00:00Z",
        "2026-09-22T08:00:06Z",
    ]
    assert body["summary"]["reference_unpaired_count"] == 1


def test_tie_on_single_pair_picks_earlier_left(client):
    # L=[0,10], R=[5]，d=10：配 l0 或 l10 时间差都是 5，字典序选 (0,0)
    post_pkg(client, _NS["ref"], 1, [
        m(1.0, "2026-09-22T08:00:00Z"),
        m(1.0, "2026-09-22T08:00:10Z"),
    ])
    post_pkg(client, _NS["tgt"], 1, [m(1.0, "2026-09-22T08:00:05Z")])
    body = compare(client).json()
    assert body["summary"]["pair_count"] == 1
    assert body["pairs"][0]["reference"]["time"] == "2026-09-22T08:00:00Z"
    assert body["summary"]["reference_unpaired_count"] == 1
    assert body["summary"]["target_unpaired_count"] == 0


def test_each_measurement_pairs_at_most_once(client):
    # 两侧各 1 条可配，时间差上限 100，也只能 1 对
    post_pkg(client, _NS["ref"], 1, [m(1.0, "2026-09-22T08:00:00Z")])
    post_pkg(client, _NS["tgt"], 1, [m(1.0, "2026-09-22T08:00:01Z")])
    body = compare(client, max_time_diff_seconds=100).json()
    assert body["summary"]["pair_count"] == 1


def test_exact_tolerance_is_not_over(client):
    post_pkg(client, _NS["ref"], 1, [m(7.0, "2026-09-22T08:00:00Z")])
    post_pkg(client, _NS["tgt"], 1, [m(7.5, "2026-09-22T08:00:00Z")])
    body = compare(client, value_tolerance=0.5).json()
    assert body["pairs"][0]["over_tolerance"] is False


def test_only_committed_packages_participate(client):
    # ref: seq 1 已提交；seq 3 因缺口 seq 2 留在缓冲区，不应参与比对
    post_pkg(client, _NS["ref"], 1, [m(1.0, "2026-09-22T08:00:00Z")])
    post_pkg(client, _NS["ref"], 3, [m(3.0, "2026-09-22T08:00:03Z")])
    post_pkg(client, _NS["tgt"], 1, [
        m(1.0, "2026-09-22T08:00:00Z"),
        m(3.0, "2026-09-22T08:00:03Z"),
    ])
    body = compare(client).json()
    assert body["summary"]["reference_candidate_count"] == 1
    assert body["summary"]["pair_count"] == 1
    # 缓冲未被比对操作改动：seq 3 仍在缓冲区
    gaps = client.get(f"/api/v1/devices/{_NS['ref']}/gaps").json()
    assert gaps["buffered_seqs"] == [3]
    assert gaps["gaps"] == [2]


def test_limit_200_per_side_rejected_not_truncated(client):
    post_pkg(client, _NS["ref"], 1, [
        m(float(i), f"2026-09-22T08:{i // 60:02d}:{i % 60:02d}Z")
        for i in range(201)
    ])
    post_pkg(client, _NS["tgt"], 1, [m(1.0, "2026-09-22T08:00:00Z")])
    r = compare(client)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "200" in detail and "拒绝" in detail
    assert "201" in detail


def test_exactly_200_per_side_accepted(client):
    post_pkg(client, _NS["ref"], 1, [
        m(float(i), f"2026-09-22T08:{(i + 1) // 60:02d}:{(i + 1) % 60:02d}Z")
        for i in range(200)
    ])
    post_pkg(client, _NS["tgt"], 1, [
        m(float(i), f"2026-09-22T08:{(i + 1) // 60:02d}:{(i + 1) % 60:02d}Z")
        for i in range(200)
    ])
    r = compare(client, max_time_diff_seconds=0)
    assert r.status_code == 200
    assert r.json()["summary"]["pair_count"] == 200


def test_unknown_device_404(client):
    post_pkg(client, _NS["ref"], 1, [m(1.0, "2026-09-22T08:00:00Z")])
    r = compare(client, ref=_NS["ref"], tgt="ghost")
    assert r.status_code == 404
    assert "ghost" in r.json()["detail"]


def test_same_device_rejected(client):
    post_pkg(client, _NS["ref"], 1, [m(1.0, "2026-09-22T08:00:00Z")])
    r = compare(client, ref=_NS["ref"], tgt=_NS["ref"])
    assert r.status_code == 422
    assert "不同" in r.json()["detail"]


def test_invalid_range_and_params(client):
    post_pkg(client, _NS["ref"], 1, [m(1.0, "2026-09-22T08:00:00Z")])
    post_pkg(client, _NS["tgt"], 1, [m(1.0, "2026-09-22T08:00:00Z")])
    r = compare(client, **{"from": "2026-09-22T09:00:00Z",
                           "to": "2026-09-22T08:00:00Z"})
    assert r.status_code == 422
    r = compare(client, max_time_diff_seconds=-1)
    assert r.status_code == 422
    # 缺少必填参数 → FastAPI 422
    r = client.get("/api/v1/metrics/compare", params={
        "reference_device": "ref", "metric": "temperature",
        "from": "2026-09-22T08:00:00Z", "to": "2026-09-22T09:00:00Z",
        "max_time_diff_seconds": 5, "value_tolerance": 0.5,
    })
    assert r.status_code == 422


def test_naive_datetimes_treated_as_utc(client):
    post_pkg(client, _NS["ref"], 1, [m(1.0, "2026-09-22T08:00:00Z")])
    post_pkg(client, _NS["tgt"], 1, [m(1.0, "2026-09-22T08:00:00Z")])
    r = compare(client, **{"from": "2026-09-22T08:00:00",
                           "to": "2026-09-22T09:00:00"})
    assert r.status_code == 200
    assert r.json()["summary"]["pair_count"] == 1


def test_does_not_modify_committed_payloads(client):
    post_pkg(client, _NS["ref"], 1, [m(20.0, "2026-09-22T08:00:00Z", unit="C")])
    post_pkg(client, _NS["tgt"], 1, [m(20.2, "2026-09-22T08:00:00Z", unit="C")])
    compare(client)
    # 重复查询结果一致，原始值未被改写
    b1 = compare(client).json()
    b2 = compare(client).json()
    assert b1 == b2
    assert b1["pairs"][0]["reference"]["value"] == 20.0
