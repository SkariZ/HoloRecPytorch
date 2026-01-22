# code/tracking/linking.py
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

from .detection import PeakDetection


@dataclass
class DetectionRecord:
    frame: int
    y: float
    x: float
    score: float


@dataclass
class TrackPoint:
    frame: int
    y: float
    x: float
    score: float


@dataclass
class Track:
    track_id: int
    points: List[TrackPoint]


def link_detections(
    detections_by_frame: List[List[PeakDetection]],
    *,
    max_link_dist: float = 10.0,
    max_frame_gap: int = 0,
    min_track_len: int = 3,
) -> List[Track]:
    """
    Multi-object linking with Hungarian assignment frame-to-frame.
    Allows short gaps via max_frame_gap.
    """
    next_id = 0
    active: dict[int, Track] = {}
    last_seen: dict[int, int] = {}
    finished: List[Track] = []

    def close_track(tid: int):
        finished.append(active[tid])
        del active[tid]
        del last_seen[tid]

    for f, dets in enumerate(detections_by_frame):
        # close stale tracks
        stale = [tid for tid, lf in last_seen.items() if f - lf > (max_frame_gap + 1)]
        for tid in stale:
            close_track(tid)

        if len(dets) == 0:
            continue

        if len(active) == 0:
            for d in dets:
                active[next_id] = Track(next_id, [TrackPoint(f, d.y, d.x, d.score)])
                last_seen[next_id] = f
                next_id += 1
            continue

        active_ids = list(active.keys())
        A = len(active_ids)
        D = len(dets)

        cost = np.full((A, D), 1e9, dtype=np.float32)

        for i, tid in enumerate(active_ids):
            last_pt = active[tid].points[-1]
            for j, d in enumerate(dets):
                dist = float(np.hypot(d.y - last_pt.y, d.x - last_pt.x))
                if dist <= max_link_dist:
                    cost[i, j] = dist

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_dets = set()
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] >= 1e8:
                continue
            tid = active_ids[i]
            d = dets[j]
            active[tid].points.append(TrackPoint(f, d.y, d.x, d.score))
            last_seen[tid] = f
            matched_dets.add(j)

        # unmatched detections -> new tracks
        for j, d in enumerate(dets):
            if j in matched_dets:
                continue
            active[next_id] = Track(next_id, [TrackPoint(f, d.y, d.x, d.score)])
            last_seen[next_id] = f
            next_id += 1

    # close remaining
    for tid in list(active.keys()):
        close_track(tid)

    finished = [t for t in finished if len(t.points) >= min_track_len]
    return finished


def flatten_detections(detections_by_frame: List[List[PeakDetection]]) -> List[DetectionRecord]:
    out: List[DetectionRecord] = []
    for f, dets in enumerate(detections_by_frame):
        for d in dets:
            out.append(DetectionRecord(f, d.y, d.x, d.score))
    return out


def save_detections_csv(path: str, dets: List[DetectionRecord]) -> None:
    with open(path, "w") as f:
        f.write("frame,y,x,score\n")
        for d in dets:
            f.write(f"{d.frame},{d.y:.3f},{d.x:.3f},{d.score:.6g}\n")


def save_tracks_csv(path: str, tracks: List[Track]) -> None:
    with open(path, "w") as f:
        f.write("track_id,frame,y,x,score\n")
        for tr in tracks:
            for p in tr.points:
                f.write(f"{tr.track_id},{p.frame},{p.y:.3f},{p.x:.3f},{p.score:.6g}\n")
