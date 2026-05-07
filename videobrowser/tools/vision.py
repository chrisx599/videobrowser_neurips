import base64
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from decord import VideoReader, cpu
except ImportError:
    print("⚠️ Decord not installed. Vision features will be disabled.")
    VideoReader = None

try:
    import av  # pyav — fallback for codecs decord can't handle (e.g. AV1)
except ImportError:
    av = None


def _encode_frame(frame_array: np.ndarray) -> str:
    """RGB (H,W,3) numpy -> base64-encoded JPEG (thumbnail 512)."""
    img = Image.fromarray(frame_array)
    img.thumbnail((512, 512))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=70)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _pyav_get_meta(container, stream) -> tuple[int, float]:
    """Return (total_frames, fps). Falls back to duration*rate if `frames` is 0."""
    fps = float(stream.average_rate) if stream.average_rate else 0.0
    total = stream.frames or 0
    if total <= 0 and stream.duration and stream.time_base and fps > 0:
        duration = float(stream.duration * stream.time_base)
        total = int(duration * fps)
    return total, fps


def _pyav_sample_indices(video_path: str, target_indices: list[int]):
    """Decode frames at the requested frame indices via pyav. Returns (frames_rgb_list, fps)."""
    if av is None:
        raise RuntimeError("pyav not installed")
    target_set = set(int(i) for i in target_indices)
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        total, fps = _pyav_get_meta(container, stream)
        results: dict[int, np.ndarray] = {}
        # Linear decode. pyav does in-container AV1/VP9 decoding; we iterate
        # until we've satisfied all target indices. For very long videos this
        # is still linear decode, but it's bounded by the highest target
        # index, which is total-1.
        for i, frame in enumerate(container.decode(stream)):
            if i in target_set:
                results[i] = frame.to_ndarray(format="rgb24")
                if len(results) >= len(target_set):
                    break
        ordered = [results[i] for i in target_indices if i in results]
    if not ordered:
        return [], fps, 0
    return ordered, fps, total


def _read_uniform_with_meta(video_path: str, num_frames: int):
    """Try decord first; fall back to pyav. Returns (frames_rgb_list, indices, fps)."""
    # First attempt: decord (fast on h264/vp9, fails on av1)
    if VideoReader is not None:
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            fps = vr.get_avg_fps()
            if total_frames > 0:
                indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
                frames_batch = vr.get_batch(indices).asnumpy()
                return [frames_batch[i] for i in range(len(indices))], list(indices), float(fps)
        except Exception:
            pass
    # Fallback: pyav
    if av is None:
        return [], [], 0.0
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        total, fps = _pyav_get_meta(container, stream)
    if total <= 0 or fps <= 0:
        return [], [], 0.0
    indices = list(np.linspace(0, total - 1, num_frames, dtype=int))
    frames, fps, _ = _pyav_sample_indices(video_path, indices)
    return frames, indices, fps


def _read_window_with_meta(video_path: str, start_time: float, end_time: float,
                           fps_sample: float, max_frames: int):
    """decord first; pyav fallback for a time window."""
    if VideoReader is not None:
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            fps = vr.get_avg_fps()
            total_frames = len(vr)
            start_frame = max(0, int(start_time * fps))
            end_frame = min(total_frames - 1, int(end_time * fps))
            if end_frame > start_frame and fps > 0:
                duration = end_time - start_time
                num_frames = max(1, min(int(duration * fps_sample), max_frames))
                indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
                frames_batch = vr.get_batch(indices).asnumpy()
                return [frames_batch[i] for i in range(len(indices))], list(indices), float(fps)
        except Exception:
            pass
    if av is None:
        return [], [], 0.0
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        total, fps = _pyav_get_meta(container, stream)
    if total <= 0 or fps <= 0:
        return [], [], 0.0
    start_frame = max(0, int(start_time * fps))
    end_frame = min(total - 1, int(end_time * fps))
    if end_frame <= start_frame:
        return [], [], fps
    duration = end_time - start_time
    num_frames = max(1, min(int(duration * fps_sample), max_frames))
    indices = list(np.linspace(start_frame, end_frame, num_frames, dtype=int))
    frames, fps, _ = _pyav_sample_indices(video_path, indices)
    return frames, indices, fps


def extract_frames_from_video(video_path: str, num_frames: int = 10):
    try:
        frames, _, _ = _read_uniform_with_meta(video_path, num_frames)
        return [_encode_frame(f) for f in frames]
    except Exception as e:
        print(f"❌ [Vision Tool] Error extracting frames: {e}")
        return []


def extract_frames_with_timestamps(video_path: str, num_frames: int = 16):
    """
    Returns list of {'timestamp': float, 'image': base64_str}.
    """
    try:
        frames, indices, fps = _read_uniform_with_meta(video_path, num_frames)
        if not frames or fps <= 0:
            return []
        return [
            {"timestamp": float(idx) / fps, "image": _encode_frame(arr)}
            for idx, arr in zip(indices, frames)
        ]
    except Exception as e:
        print(f"❌ [JIT Vision] Error extracting frames: {e}")
        return []


def _load_grid_font(size: int = 22):
    """Try a few common system fonts; fall back to PIL default if none exist."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size=size)
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _draw_timestamp_label(draw: "ImageDraw.ImageDraw", x: int, y: int, text: str, font):
    """Draw white text with a 1-px black outline so it's readable on any background."""
    if font is None:
        return
    for dx, dy in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        draw.text((x + dx, y + dy), text, fill="black", font=font)
    draw.text((x, y), text, fill="white", font=font)


def extract_frames_as_grids(
    video_path: str,
    num_grids: int = 16,
    frames_per_grid: int = 4,
    cell_size: tuple = (384, 384),
):
    """Sample `num_grids * frames_per_grid` frames uniformly across the video, then
    pack each consecutive group of `frames_per_grid` frames into a 2x2 mosaic image
    with each cell labelled by its timestamp.

    Returns a list of dicts, one per grid:
        [{
            "image": base64_jpeg,
            "timestamps": [t1, t2, t3, t4],
            "grid_index": int,
            "layout": (rows, cols),
        }, ...]

    Cell ordering inside each grid is left-to-right, top-to-bottom (TL, TR, BL, BR).
    Token cost vs `extract_frames_with_timestamps`: same number of image_urls
    (num_grids), but each grid carries `frames_per_grid` × the temporal coverage.
    """
    _LAYOUTS = {4: (2, 2), 9: (3, 3), 16: (4, 4)}
    if frames_per_grid not in _LAYOUTS:
        raise ValueError(
            f"frames_per_grid must be one of {sorted(_LAYOUTS)}; got {frames_per_grid}"
        )
    layout = _LAYOUTS[frames_per_grid]
    rows, cols = layout
    cell_w, cell_h = cell_size
    grid_w, grid_h = cell_w * cols, cell_h * rows

    total_to_sample = num_grids * frames_per_grid

    try:
        frames, indices, fps = _read_uniform_with_meta(video_path, total_to_sample)
    except Exception as e:
        print(f"❌ [Vision Tool] Grid extraction failed: {e}")
        return []
    if not frames or fps <= 0:
        return []

    # If fewer frames available than requested, pad with the last frame to keep
    # the grid count stable. This is rare (very short clips).
    if len(frames) < total_to_sample:
        last = frames[-1]
        last_idx = indices[-1]
        while len(frames) < total_to_sample:
            frames.append(last)
            indices.append(last_idx)

    font = _load_grid_font(size=max(16, cell_h // 18))

    grids = []
    for g in range(num_grids):
        canvas = Image.new("RGB", (grid_w, grid_h), color="black")
        draw = ImageDraw.Draw(canvas)
        timestamps = []
        for k in range(frames_per_grid):
            i = g * frames_per_grid + k
            cell_img = Image.fromarray(frames[i]).resize(cell_size, Image.BILINEAR)
            r = k // cols
            c = k % cols
            x0 = c * cell_w
            y0 = r * cell_h
            canvas.paste(cell_img, (x0, y0))
            ts = float(indices[i]) / fps
            timestamps.append(ts)
            _draw_timestamp_label(
                draw, x0 + 6, y0 + 4, f"t={ts:.1f}s", font
            )

        buffer = io.BytesIO()
        canvas.save(buffer, format="JPEG", quality=75)
        grids.append({
            "image": base64.b64encode(buffer.getvalue()).decode("utf-8"),
            "timestamps": timestamps,
            "grid_index": g,
            "layout": layout,
        })

    return grids


def extract_frames_from_window(video_path: str, start_time: float, end_time: float,
                               fps_sample: float = 1.0, max_frames: int = 32):
    """
    Returns list of {'timestamp': float, 'image': base64_str} sampled inside [start_time, end_time].
    """
    try:
        frames, indices, fps = _read_window_with_meta(
            video_path, start_time, end_time, fps_sample, max_frames
        )
        if not frames or fps <= 0:
            return []
        return [
            {"timestamp": float(idx) / fps, "image": _encode_frame(arr)}
            for idx, arr in zip(indices, frames)
        ]
    except Exception as e:
        print(f"❌ [Vision Tool] Error extracting window frames: {e}")
        return []
