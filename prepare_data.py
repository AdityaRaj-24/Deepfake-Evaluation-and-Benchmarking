"""
prepare_data.py
===============
Converts the Kaggle FaceForensics++ C23 dataset (MP4 videos) into a
clean train / val / test image dataset for deepfake detection.

Kaggle dataset: https://www.kaggle.com/datasets/xdxd003/ff-c23

Expected input layout
---------------------
<ff_root>/
    original/                        ← real videos
        <video_id>.mp4
    Deepfakes/
        <video_id>.mp4
    FaceSwap/
        <video_id>.mp4
    FaceShifter/
        <video_id>.mp4

Output layout
-------------
<out_dir>/
    dataset/
        train/
            real/
            fake/
                Deepfakes/
                FaceSwap/
                FaceShifter/
        val/
            real/
            fake/
        test/
            real/
            fake/
    metadata.csv

metadata.csv columns
--------------------
filename, label (0=real/1=fake), manipulation, video_id, split

Usage (quick-start)
-------------------
    # Fast mode – resize frames only, no face-detection
    python prepare_data.py --ff_root /data/ff_c23 --no_align

    # Full mode – face-detect + align every frame
    python prepare_data.py --ff_root /data/ff_c23 \\
        --frames_per_video 30 --face_margin 0.20

    # Use frame stride instead of a fixed count
    python prepare_data.py --ff_root /data/ff_c23 --frame_stride 5

    # Use all four manipulation types
    python prepare_data.py --ff_root /data/ff_c23 \\
        --methods Deepfakes FaceSwap FaceShifter

Notes
-----
* The train / val / test split is done at VIDEO level before any frame
  is extracted, preventing temporal data leakage between splits.
* Face detection uses RetinaFace (dlib HOG).  Install with:
      pip install RetinaFace
  If the library is absent the script automatically falls back to
  plain resize (equivalent to --no_align).
* All randomness is seeded so runs are fully reproducible.
"""

from __future__ import annotations
import time
import argparse
import csv
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
from PIL import Image

# ── optional face-detection dependency ────────────────────────────────────────
try:
    from retinaface import RetinaFace        # type: ignore
    import numpy as np
    _HAS_RF = True
except ImportError:
    _HAS_RF = False
    import numpy as np               # always available via cv2

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """All run-time parameters in one place."""
    ff_root:           Path
    out_dir:           Path
    methods:           list[str]
    train_ratio:       float
    val_ratio:         float
    # frame extraction
    frames_per_video:  int | None   # None → use stride instead
    frame_stride:      int          # used only when frames_per_video is None
    # face detection
    align:             bool
    face_margin:       float
    face_size:         int
    # reproducibility
    seed:              int
    # dataset size
    videos_per_method: int | None   
    # book-keeping
    dataset_dir:       Path = field(init=False)
    meta_path:         Path = field(init=False)

    def __post_init__(self) -> None:
        self.dataset_dir = self.out_dir / "dataset"
        self.meta_path   = self.out_dir / "metadata.csv"


@dataclass
class VideoRecord:
    video_id:     str
    path:         Path
    manipulation: str      # "real" or e.g. "Deepfakes"
    label:        int      # 0 = real, 1 = fake
    split:        str      # "train" | "val" | "test"


@dataclass
class FrameRecord:
    """One saved image frame."""
    filename:     str      # relative to dataset_dir
    label:        int
    manipulation: str
    video_id:     str
    split:        str


# ══════════════════════════════════════════════════════════════════════════════
# Video discovery
# ══════════════════════════════════════════════════════════════════════════════

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def _find_videos(folder: Path, manipulation: str, label: int) -> list[VideoRecord]:
    """Return all MP4-like files under *folder* as VideoRecord objects."""
    if not folder.exists():
        log.warning("Folder not found, skipping: %s", folder)
        return []
    records = []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in _VIDEO_EXTS:
            records.append(VideoRecord(
                video_id     = p.stem,
                path         = p,
                manipulation = manipulation,
                label        = label,
                split        = "",           # filled in during split step
            ))
    log.info("  %-14s → %4d videos", manipulation, len(records))
    return records


def discover_videos(cfg: Config) -> dict[str, list[VideoRecord]]:
    """
    Return a dict keyed by manipulation type (incl. "real").
    Each value is a randomly-shuffled list of VideoRecord objects.
    """
    log.info("Discovering videos under %s", cfg.ff_root)
    rng = random.Random(cfg.seed)

    # Real
    real_dir = cfg.ff_root / "original"
    by_type: dict[str, list[VideoRecord]] = {}
    real_vids = _find_videos(real_dir, manipulation="real", label=0)
    rng.shuffle(real_vids)

    if cfg.videos_per_method is not None:
        real_vids = real_vids[:cfg.videos_per_method]

    by_type["real"] = real_vids

    # Fake
    for method in cfg.methods:
        fake_dir = cfg.ff_root / method
        vids = _find_videos(fake_dir, manipulation=method, label=1)
        rng.shuffle(vids)

        if cfg.videos_per_method is not None:
            vids = vids[:cfg.videos_per_method]

        by_type[method] = vids

    total = sum(len(v) for v in by_type.values())
    log.info("Total videos discovered: %d", total)
    return by_type


# ══════════════════════════════════════════════════════════════════════════════
# Video-level train / val / test split
# ══════════════════════════════════════════════════════════════════════════════

def split_videos(
    by_type: dict[str, list[VideoRecord]],
    cfg: Config,
) -> list[VideoRecord]:
    """
    Assign each video a split label.  The split is done per manipulation
    type to keep the real/fake ratio consistent across splits.
    Returns a flat list of all VideoRecord objects (with split filled in).
    """
    test_ratio = 1.0 - cfg.train_ratio - cfg.val_ratio
    assert test_ratio > 0, "train_ratio + val_ratio must be < 1.0"

    all_records: list[VideoRecord] = []

    for manip, videos in by_type.items():
        n = len(videos)
        n_train = int(n * cfg.train_ratio)
        n_val   = int(n * cfg.val_ratio)
        # test gets the remainder so rounding never loses a video
        for i, v in enumerate(videos):
            if i < n_train:
                v.split = "train"
            elif i < n_train + n_val:
                v.split = "val"
            else:
                v.split = "test"
        train_c = sum(1 for v in videos if v.split == "train")
        val_c   = sum(1 for v in videos if v.split == "val")
        test_c  = sum(1 for v in videos if v.split == "test")
        log.info("  %-14s  train=%d  val=%d  test=%d", manip, train_c, val_c, test_c)
        all_records.extend(videos)

    return all_records


# ══════════════════════════════════════════════════════════════════════════════
# Frame extraction helpers
# ══════════════════════════════════════════════════════════════════════════════

def _frame_indices(total_frames: int, cfg: Config) -> list[int]:
    """
    Choose which frame indices to extract from a video.

    Two modes:
      frames_per_video  → evenly spaced across the video
      frame_stride      → every N-th frame starting from 0
    """
    if total_frames <= 0:
        return []

    if cfg.frames_per_video is not None:
        k = min(cfg.frames_per_video, total_frames)
        if k == 1:
            return [total_frames // 2]
        step = max(1, (total_frames - 1) // (k - 1))
        indices = list(range(0, total_frames, step))[:k]
    else:
        indices = list(range(0, total_frames, max(1, cfg.frame_stride)))

    return indices


def _read_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    """Seek to *idx* and return a BGR numpy frame, or None on failure."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None


# ══════════════════════════════════════════════════════════════════════════════
# Face detection / cropping
# ══════════════════════════════════════════════════════════════════════════════

def _crop_face_fr(
    rgb: np.ndarray,
    margin: float,
    size: int,
) -> Image.Image | None:
    """
    Detect the largest face using RetinaFace.
    Crop with margin and resize.
    """

    try:
        detections = RetinaFace.detect_faces(rgb)
    except Exception:
        return None

    if not detections:
        return None

    best_face = None
    best_area = 0

    for _, det in detections.items():

        x1, y1, x2, y2 = det["facial_area"]

        area = (x2 - x1) * (y2 - y1)

        if area > best_area:
            best_area = area
            best_face = (x1, y1, x2, y2)

    if best_face is None:
        return None

    x1, y1, x2, y2 = best_face

    h, w = rgb.shape[:2]

    pad = int(max(x2 - x1, y2 - y1) * margin)

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    face = rgb[y1:y2, x1:x2]

    if face.size == 0:
        return None

    return Image.fromarray(face).resize(
        (size, size),
        Image.LANCZOS,
    )

def _resize_only(rgb: np.ndarray, size: int) -> Image.Image:
    """Fallback: just resize the full frame, no face detection."""
    return Image.fromarray(rgb).resize((size, size), Image.LANCZOS)


def process_frame(
    bgr: np.ndarray,
    cfg: Config,
) -> Image.Image | None:
    """
    Convert a BGR OpenCV frame to a face-cropped (or resized) PIL image.
    Returns None if face-detection is requested but no face was found.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if cfg.align and _HAS_RF:
        return _crop_face_fr(rgb, cfg.face_margin, cfg.face_size)
    return _resize_only(rgb, cfg.face_size)


# ══════════════════════════════════════════════════════════════════════════════
# Output path construction
# ══════════════════════════════════════════════════════════════════════════════

def _dest_dir(cfg: Config, video: VideoRecord) -> Path:
    split = video.split

    if video.label == 0:
        return cfg.dataset_dir / split / "real"

    return cfg.dataset_dir / split / "fake" / video.manipulation


# ══════════════════════════════════════════════════════════════════════════════
# Main extraction loop
# ══════════════════════════════════════════════════════════════════════════════

def extract_all(
    all_videos: list[VideoRecord],
    cfg: Config,
) -> list[FrameRecord]:
    """
    Iterate over every video, extract frames, save them, and return
    a list of FrameRecord objects for the metadata CSV.
    """
    if cfg.align and not _HAS_RF:
        log.warning(
            "RetinaFace not installed – falling back to resize-only mode. "
            "Install with:  pip install RetinaFace"
        )

    frame_records: list[FrameRecord] = []
    total = len(all_videos)
    start_time = time.time()

    for vid_idx, video in enumerate(all_videos, 1):
        log.info(
            "[%d/%d] %s  %-14s  split=%s",
            vid_idx, total, video.video_id, video.manipulation, video.split,
        )

        cap = cv2.VideoCapture(str(video.path))
        if not cap.isOpened():
            log.warning("  Could not open %s – skipping.", video.path)
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = _frame_indices(total_frames, cfg)

        dest = _dest_dir(cfg, video)
        dest.mkdir(parents=True, exist_ok=True)

        saved = 0
        skipped = 0
        for frame_idx in indices:
            bgr = _read_frame(cap, frame_idx)
            if bgr is None:
                skipped += 1
                continue

            img = process_frame(bgr, cfg)
            if img is None:               # no face detected
                skipped += 1
                continue

            fname = f"{video.video_id}_f{frame_idx:06d}.jpg"
            fpath = dest / fname

            img.save(
                fpath,
                format="JPEG",
                quality=95,
            )

            # Store relative path (relative to dataset_dir) for portability
            rel = str(fpath.relative_to(cfg.dataset_dir))
            frame_records.append(FrameRecord(
                filename     = rel,
                label        = video.label,
                manipulation = video.manipulation,
                video_id     = video.video_id,
                split        = video.split,
            ))
            saved += 1

        cap.release()
        log.info("  → saved %d frames  (skipped %d)", saved, skipped)
        # Progress report every 10 videos (and at the end)
        if vid_idx % 20 == 0 or vid_idx == total:

            elapsed = time.time() - start_time

            avg_per_video = elapsed / vid_idx

            remaining = avg_per_video * (total - vid_idx)

            hrs_e = int(elapsed // 3600)
            mins_e = int((elapsed % 3600) // 60)

            hrs_r = int(remaining // 3600)
            mins_r = int((remaining % 3600) // 60)

            log.info(
                "Processed %d/%d videos (%.1f%%)",
                vid_idx,
                total,
                100 * vid_idx / total,
            )

            log.info(
                "Elapsed: %dh %02dm",
                hrs_e,
                mins_e,
            )

            log.info(
                "Estimated remaining: %dh %02dm",
                hrs_r,
                mins_r,
            )

    return frame_records


# ══════════════════════════════════════════════════════════════════════════════
# Metadata CSV
# ══════════════════════════════════════════════════════════════════════════════

_CSV_FIELDS = ["filename", "label", "manipulation", "video_id", "split"]


def write_metadata(records: list[FrameRecord], path: Path) -> None:
    """Write metadata.csv to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "filename":     r.filename,
                "label":        r.label,
                "manipulation": r.manipulation,
                "video_id":     r.video_id,
                "split":        r.split,
            })
    log.info("Metadata written → %s  (%d rows)", path, len(records))


def print_summary(records: list[FrameRecord]) -> None:
    """Print a split × manipulation frame count table."""
    from collections import defaultdict
    tbl: dict[tuple[str, str], int] = defaultdict(int)
    for r in records:
        tbl[(r.split, r.manipulation)] += 1

    splits = ["train", "val", "test"]
    manipulations = sorted({r.manipulation for r in records})

    col_w = max(14, max(len(m) for m in manipulations) + 2)
    header = f"{'':10}" + "".join(f"{m:>{col_w}}" for m in manipulations) + f"{'TOTAL':>10}"
    log.info("\n%s", header)
    log.info("%s", "-" * len(header))

    for sp in splits:
        row_total = sum(tbl[(sp, m)] for m in manipulations)
        row = f"{sp:<10}" + "".join(
            f"{tbl[(sp, m)]:>{col_w}}" for m in manipulations
        ) + f"{row_total:>10}"
        log.info(row)

    grand = sum(tbl.values())
    log.info("%s", "-" * len(header))
    log.info(
        "%s", f"{'TOTAL':<10}"
              + "".join(f"{sum(tbl[(s, m)] for s in splits):>{col_w}}" for m in manipulations)
              + f"{grand:>10}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args(argv: list[str] | None = None) -> Config:
    ap = argparse.ArgumentParser(
        description="Prepare FaceForensics++ C23 dataset for deepfake detection training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── paths ─────────────────────────────────────────────────────────────────
    ap.add_argument(
        "--ff_root", required=True, type=Path,
        help="Root folder containing original/, Deepfakes/, FaceSwap/, FaceShifter/",
    )
    ap.add_argument(
        "--out_dir", default=Path("."), type=Path,
        help="Output root.  dataset/ and metadata.csv are written here.",
    )

    # ── manipulation types ────────────────────────────────────────────────────
    ap.add_argument(
        "--methods", nargs="+",
        default=["Deepfakes", "FaceSwap", "FaceShifter"],
        choices=["Deepfakes", "FaceSwap", "FaceShifter", "Face2Face", "NeuralTextures"],
        help="Which fake manipulation subfolder(s) to include.",
    )
    ap.add_argument(
        "--videos_per_method",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of videos to use from each manipulation folder.",
    )
    # ── split ratios ──────────────────────────────────────────────────────────
    ap.add_argument("--train_ratio", type=float, default=0.70,
                    help="Fraction of videos for training.")
    ap.add_argument("--val_ratio",   type=float, default=0.15,
                    help="Fraction of videos for validation.  "
                         "Remainder goes to test.")

    # ── frame extraction ──────────────────────────────────────────────────────
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument(
        "--frames_per_video", type=int, default=None,
        metavar="N",
        help="Extract exactly N evenly-spaced frames per video.",
    )
    grp.add_argument(
        "--frame_stride", type=int, default=10,
        metavar="S",
        help="Extract every S-th frame (used when --frames_per_video is not set).",
    )

    # ── face detection ────────────────────────────────────────────────────────
    ap.add_argument(
        "--no_align", action="store_true",
        help="Skip face detection and just resize full frames to --face_size.",
    )
    ap.add_argument(
        "--face_margin", type=float, default=0.20,
        metavar="M",
        help="Fractional margin added around the detected face bounding box.",
    )
    ap.add_argument(
        "--face_size", type=int, default=256,
        metavar="PX",
        help="Output image size (square, pixels).",
    )

    # ── reproducibility ───────────────────────────────────────────────────────
    ap.add_argument("--seed", type=int, default=42, help="Global random seed.")

    # ── verbosity ─────────────────────────────────────────────────────────────
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Set logging level to DEBUG.")

    args = ap.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate ratios
    if not (0 < args.train_ratio < 1):
        ap.error("--train_ratio must be in (0, 1).")
    if not (0 < args.val_ratio < 1):
        ap.error("--val_ratio must be in (0, 1).")
    if args.train_ratio + args.val_ratio >= 1.0:
        ap.error("--train_ratio + --val_ratio must be < 1.0 (test gets the rest).")

    return Config(
        ff_root          = args.ff_root.expanduser().resolve(),
        out_dir          = args.out_dir.expanduser().resolve(),
        methods          = args.methods,
        train_ratio      = args.train_ratio,
        val_ratio        = args.val_ratio,
        frames_per_video = args.frames_per_video,
        frame_stride     = args.frame_stride,
        align            = not args.no_align,
        face_margin      = args.face_margin,
        face_size        = args.face_size,
        seed             = args.seed,
        videos_per_method = args.videos_per_method,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def run(cfg: Config) -> list[FrameRecord]:
    """Full pipeline.  Returns the list of FrameRecord objects written."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    log.info("FaceForensics++ C23 dataset preparation")
    log.info("  ff_root  : %s", cfg.ff_root)
    log.info("  out_dir  : %s", cfg.out_dir)
    log.info("  methods  : %s", cfg.methods)
    log.info("  align    : %s%s",
             cfg.align,
             "" if _HAS_RF or not cfg.align else " (RetinaFace unavailable → resize only)")
    log.info(
        "  frames   : %s",
        f"per_video={cfg.frames_per_video}"
        if cfg.frames_per_video
        else f"stride={cfg.frame_stride}",
    )

    # 1. Discover videos
    by_type = discover_videos(cfg)

    # 2. Split at video level
    log.info("Splitting videos (train=%.0f%% / val=%.0f%% / test=%.0f%%)…",
             cfg.train_ratio * 100, cfg.val_ratio * 100,
             (1 - cfg.train_ratio - cfg.val_ratio) * 100)
    all_videos = split_videos(by_type, cfg)

    # 3. Extract frames
    log.info("Extracting frames…")
    records = extract_all(all_videos, cfg)

    # 4. Write metadata
    write_metadata(records, cfg.meta_path)

    # 5. Print summary table
    print_summary(records)

    log.info("✅  Done.  Dataset at: %s", cfg.dataset_dir)
    return records


def main(argv: list[str] | None = None) -> None:
    cfg = _parse_args(argv)
    run(cfg)


if __name__ == "__main__":
    main()