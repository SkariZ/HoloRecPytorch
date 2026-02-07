import numpy as np

def load_analysis_index(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    out = {
        "track_id": data["track_id"].astype(np.int32),
        "t": data["t"].astype(np.int32),
        "x": data["x"].astype(np.float32),
        "y": data["y"].astype(np.float32),
        "score": data["score"].astype(np.float32) if "score" in data.files else None,
        "input_path": str(data["input_path"]) if "input_path" in data.files else None,
        "roi_radius": int(data["roi_radius"]) if "roi_radius" in data.files else None,
        "channel": str(data["channel"]) if "channel" in data.files else None,
    }
    return out


def group_by_track(track_id: np.ndarray) -> list[tuple[int, np.ndarray]]:
    track_id = track_id.astype(np.int32, copy=False)
    order = np.argsort(track_id, kind="stable")
    tid_sorted = track_id[order]

    if tid_sorted.size == 0:
        return []

    cuts = np.flatnonzero(np.diff(tid_sorted)) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, tid_sorted.size]

    groups = []
    for s, e in zip(starts, ends):
        tid = int(tid_sorted[s])
        idx = order[s:e]
        groups.append((tid, idx))
    return groups
