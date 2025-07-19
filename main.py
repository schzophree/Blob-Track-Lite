from __future__ import annotations

import logging
import os
import random
import tempfile
import warnings
from pathlib import Path

import cv2
import librosa
import moviepy.editor as mpy
import numpy as np
import ffmpeg
import string

# Suppress only specific MoviePy warning
warnings.filterwarnings(
    "ignore",
    message="Warning: in file .* bytes wanted but 0 bytes read.*",
    category=UserWarning,
    module="moviepy.video.io.ffmpeg_reader"
)


def _extract_audio(video_path: Path, sr: int = 22050) -> Path:
    tmp_dir = Path(tempfile.mkdtemp())
    wav_path = tmp_dir / "temp_audio.wav"
    clip = mpy.VideoFileClip(str(video_path))
    clip.audio.write_audiofile(str(wav_path), fps=sr, logger=None, verbose=False)
    return wav_path


def _detect_onsets(wav_path: Path, sr: int = 22050) -> np.ndarray:
    y, _ = librosa.load(str(wav_path), sr=sr)
    return librosa.onset.onset_detect(y=y, sr=sr, units="time")


def get_rotation(path: Path) -> int:
    try:
        meta = ffmpeg.probe(str(path))
        for stream in meta["streams"]:
            if stream["codec_type"] == "video" and "tags" in stream and "rotate" in stream["tags"]:
                return int(stream["tags"]["rotate"])
    except Exception:
        return 0
    return 0


class TrackedPoint:
    def __init__(self, pos: tuple[float, float], life: int, size: int):
        self.pos = np.array(pos, dtype=np.float32)
        self.life = life
        self.size = size


def _sample_size_bell(min_s: int, max_s: int, width_div: float = 6.0) -> int:
    mean = (min_s + max_s) / 2.0
    sigma = (max_s - min_s) / width_div
    for _ in range(10):
        val = np.random.normal(mean, sigma)
        if min_s <= val <= max_s:
            return int(val)
    return int(np.clip(val, min_s, max_s))

def _generate_random_text(length: int = 8) -> str:
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))


def render_tracked_effect(
    video_in: Path,
    video_out: Path,
    *,
    fps: float | None,
    pts_per_beat: int,
    ambient_rate: float,
    jitter_px: float,
    life_frames: int,
    min_size: int,
    max_size: int,
    neighbor_links: int,
    orb_fast_threshold: int,
    bell_width: float,
    seed: int | None,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    rotation = get_rotation(video_in)
    clip = mpy.VideoFileClip(str(video_in))

    if rotation == 90:
        clip = clip.rotate(-90)
    elif rotation == 270:
        clip = clip.rotate(90)
    elif rotation == 180:
        clip = clip.rotate(180)

    fps = fps or clip.fps
    clip = clip.set_duration(clip.duration)
    frame_w, frame_h = clip.size

    wav_path = _extract_audio(video_in)
    onset_times = _detect_onsets(wav_path)
    logging.info("%d onsets detected", len(onset_times))

    orb = cv2.ORB_create(nfeatures=1500, fastThreshold=orb_fast_threshold)
    active: list[TrackedPoint] = []
    onset_idx = 0
    prev_gray: np.ndarray | None = None

    def make_frame(t: float):
        nonlocal prev_gray, onset_idx, active
        frame = clip.get_frame(t).copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        if prev_gray is not None and active:
            prev_pts = np.array([p.pos for p in active], dtype=np.float32).reshape(-1, 1, 2)
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
            new_active: list[TrackedPoint] = []
            for tp, new_pt, ok in zip(active, next_pts.reshape(-1, 2), status.reshape(-1)):
                if not ok:
                    continue
                x, y = new_pt
                if 0 <= x < w and 0 <= y < h and tp.life > 0:
                    tp.pos = new_pt
                    tp.life -= 1
                    if jitter_px > 0:
                        tp.pos += np.random.normal(0, jitter_px, size=2)
                        tp.pos = np.clip(tp.pos, [0, 0], [w - 1, h - 1])
                    new_active.append(tp)
            active = new_active

        while onset_idx < len(onset_times) and t >= onset_times[onset_idx]:
            kps = orb.detect(gray, None)
            kps = sorted(kps, key=lambda k: k.response, reverse=True)
            target_spawn = random.randint(1, pts_per_beat)
            spawned = 0
            for kp in kps:
                if spawned >= target_spawn:
                    break
                x, y = kp.pt
                if any(np.linalg.norm(tp.pos - (x, y)) < 10 for tp in active):
                    continue
                size = _sample_size_bell(min_size, max_size, bell_width)
                active.append(TrackedPoint((x, y), life_frames, size))
                spawned += 1
            onset_idx += 1

        if ambient_rate > 0:
            noise_n = np.random.poisson(ambient_rate / fps)
            for _ in range(noise_n):
                x = random.uniform(0, w)
                y = random.uniform(0, h)
                size = _sample_size_bell(min_size, max_size, bell_width)
                active.append(TrackedPoint((x, y), life_frames, size))

        coords = [tp.pos for tp in active]
        for i, p in enumerate(coords):
            dists = [(j, np.linalg.norm(p - coords[j])) for j in range(len(coords)) if j != i]
            dists.sort(key=lambda x: x[1])
            for j, _ in dists[:neighbor_links]:
                cv2.line(frame, tuple(p.astype(int)), tuple(coords[j].astype(int)), (200, 200, 255), 1)

        for tp in active:
            x, y = tp.pos
            s = tp.size
            tl = (max(0, int(x - s // 2)), max(0, int(y - s // 2)))
            br = (min(w - 1, int(x + s // 2)), min(h - 1, int(y + s // 2)))
            roi = frame[tl[1]:br[1], tl[0]:br[0]]
            if roi.size:
                frame[tl[1]:br[1], tl[0]:br[0]] = 255 - roi
            cv2.rectangle(frame, tl, br, (200, 200, 255), 1)

            text = _generate_random_text()
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_thickness = 1
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = int(x + s // 2 + 5)
            text_y = int(y - s // 2 - 5)
            # Psa
            if 0 <= text_x < w and 0 <= text_y >= text_size[1]:
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, (200, 200, 255), font_thickness, cv2.LINE_AA)

        prev_gray = gray
        return frame

    out_clip = mpy.VideoClip(make_frame, duration=clip.duration)
    out_clip = out_clip.set_audio(clip.audio).set_fps(fps).resize(clip.size)
    out_clip.write_videofile(str(video_out), codec="libx264", audio_codec="aac")


def funny_loading_bar():
    import time
    import sys

    stages = [
        "Initializing...",
        "Summoning pixels...",
        "Whispering to the GPU...",
        "Feeding bits...",
        "Untangling frames...",
        "Convincing codecs...",
        "Drawing boxes like Picasso...",
        "Launching frame...",
        "Synchronizing beats...",
        "Finishing touches...",
    ]

    for i in range(0, 101):
        time.sleep(0.03 if i < 90 else 0.01)
        bar = ('█' * (i // 2)).ljust(50)
        # ngacak ngawur untuk indexing
        stage_text = stages[i // 10 % len(stages)] if i % 10 == 0 else ''
        sys.stdout.write(
            f"\r[{bar}] {i}%  {stage_text}    "
        )
        sys.stdout.flush()
    print("\nRender prep complete!")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    input_path = input("Enter the input video file path: ").strip()
    if not os.path.isfile(input_path):
        print("❌ Error: Input file does not exist.")
        return

    default_output = Path.home() / "Downloads" / "output.mp4"
    prompt = f"Enter output video file path (leave blank to save in Downloads as '{default_output.name}'): "
    output_path_str = input(prompt).strip()
    output_path = Path(output_path_str) if output_path_str else default_output

    print("\nStarting rendering process...\n")
    funny_loading_bar()

    render_tracked_effect(
        video_in=Path(input_path),
        video_out=output_path,
        fps=None,
        pts_per_beat=20,
        ambient_rate=5.0,
        jitter_px=0.5,
        life_frames=10,
        min_size=15,
        max_size=40,
        neighbor_links=3,
        orb_fast_threshold=20,
        bell_width=4.0,
        seed=None,
    )


if __name__ == "__main__":
    main()
