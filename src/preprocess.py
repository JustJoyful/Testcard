"""Shared timestamp-aware video probing and in-memory preprocessing."""
import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np


def probe_video(path: Path, *, timestamps: bool = True) -> dict:
    path = Path(path)
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_streams', '-show_format',
               '-show_entries',
               'stream=width,height,avg_frame_rate,duration:stream_tags=rotate:'
               'stream_side_data=rotation:format=duration'
               + (':frame=best_effort_timestamp_time' if timestamps else ''), '-of', 'json', str(path)]
    if timestamps:
        command.insert(7, '-show_frames')
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=180, check=True)
        if result.stderr.strip():
            raise ValueError(result.stderr.strip())
        data = json.loads(result.stdout)
        stream = data['streams'][0]
        duration = float(stream.get('duration', data.get('format', {}).get('duration', 'nan')))
        fps = float(Fraction(stream['avg_frame_rate']))
        rotation = float(stream.get('tags', {}).get('rotate', 0))
        for side in stream.get('side_data_list', []):
            if 'rotation' in side:
                rotation = float(side['rotation'])
        if not math.isfinite(rotation) or rotation % 90 != 0:
            raise ValueError(f'unsupported rotation {rotation}')
        pts = np.array([float(f['best_effort_timestamp_time']) for f in data.get('frames', [])])
        if not math.isfinite(duration) or duration <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError('invalid duration or frame rate')
        if timestamps and (not len(pts) or not np.isfinite(pts).all() or np.any(np.diff(pts) < 0)):
            raise ValueError('missing or invalid presentation timestamps')
        # Timestamp targets are relative to the first presented frame.
        if timestamps:
            pts = pts - pts[0]
        metadata = {'duration_s': duration, 'fps': fps, 'width': int(stream['width']),
                'height': int(stream['height']), 'rotation_applied': int(rotation) % 360,
                **({'pts': pts} if timestamps else {})}
        return metadata
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError, ZeroDivisionError) as exc:
        detail = getattr(exc, 'stderr', '') or str(exc)
        raise ValueError(f'{path.name}: ffprobe failed: {detail}') from exc


def nearest_indices(pts: np.ndarray, targets: np.ndarray) -> np.ndarray:
    pts, targets = np.asarray(pts, dtype=float), np.asarray(targets, dtype=float)
    if (pts.ndim != 1 or targets.ndim != 1 or not len(pts)
            or not np.isfinite(pts).all() or not np.isfinite(targets).all()
            or np.any(np.diff(pts) < 0)):
        raise ValueError('missing or invalid monotonic presentation timestamps/targets')
    right = np.searchsorted(pts, targets).clip(0, len(pts)-1)
    left = (right-1).clip(0)
    tolerance = 8 * np.finfo(float).eps * np.maximum(1, np.abs(targets))
    indices = np.where(np.abs(targets-pts[left]) <= np.abs(pts[right]-targets)+tolerance, left, right)
    return np.searchsorted(pts, pts[indices], side='left')


def rotate_frame(frame: np.ndarray, rotation: int) -> np.ndarray:
    if rotation not in (0, 90, 180, 270):
        raise ValueError(f'unsupported rotation {rotation}')
    return np.ascontiguousarray(np.rot90(frame, rotation // 90))


def iter_samples(path: Path, config: dict):
    """Decode once; retain only frames needed by overlapping motion windows."""
    import tempfile
    path = Path(path)
    meta = probe_video(path)
    pts = meta['pts']
    motion = config.get('backbones', {}).get('motion', {}).get('candidate', 'slowfast_r50')
    count, step = (16, 2/30) if motion == 'r3d18' else (32, 2/30)
    offsets = (np.arange(count) - (count-1)/2) * step
    frame_bytes = meta['width'] * meta['height'] * 3
    command = ['ffmpeg', '-v', 'error', '-xerror', '-err_detect', 'explode', '-noautorotate',
               '-i', str(path), '-map', '0:v:0', '-an', '-sn', '-fps_mode', 'passthrough',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        decoded = 0
        frames = {}
        def read_frame():
            nonlocal decoded
            raw = process.stdout.read(frame_bytes)
            if len(raw) != frame_bytes:
                raise ValueError(f'{path.name}: truncated decode at frame {decoded}')
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(meta['height'], meta['width'], 3)
            frames[decoded] = rotate_frame(frame, meta['rotation_applied'])
            decoded += 1
        try:
            for timestamp in np.arange(0, meta['duration_s'], .25):
                motion_indices = nearest_indices(pts, timestamp + offsets)
                spatial_index = int(nearest_indices(pts, np.array([timestamp]))[0])
                needed = np.append(motion_indices, spatial_index)
                while decoded <= needed.max():
                    read_frame()
                yield float(timestamp), frames[spatial_index], np.stack([frames[int(i)] for i in motion_indices])
                for index in list(frames):
                    if index < needed.min():
                        del frames[index]
            # Validate even frames after the last sampled timestamp.
            while decoded < len(pts):
                read_frame()
                frames.clear()
            if process.stdout.read(1):
                raise ValueError(f'{path.name}: decoder/probe frame count mismatch')
            code = process.wait(timeout=60)
            errors.seek(0)
            detail = errors.read().decode(errors='replace').strip()
            if code or detail:
                raise ValueError(f'{path.name}: ffmpeg failed ({code}): {detail}')
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait()


def spatial_tensor(frame: np.ndarray):
    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as f
    tensor = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float() / 255
    tensor = f.resize(tensor, 448, interpolation=InterpolationMode.BICUBIC, antialias=True)
    return f.normalize(f.center_crop(tensor, [448, 448]), [.485, .456, .406], [.229, .224, .225])


def motion_tensors(frames: np.ndarray, config: dict):
    import torch
    from torchvision.transforms import functional as f
    if config.get('backbones', {}).get('motion', {}).get('candidate') == 'r3d18':
        from torchvision.models.video import R3D_18_Weights
        return R3D_18_Weights.KINETICS400_V1.transforms()(torch.from_numpy(frames.copy()).permute(0, 3, 1, 2))
    if frames.shape[0] != 32:
        raise ValueError('SlowFast requires 32 frames')
    tensor = torch.from_numpy(np.ascontiguousarray(frames)).permute(3, 0, 1, 2).float() / 255
    tensor = (tensor - .45) / .225
    height, width = tensor.shape[-2:]
    size = (256, math.floor(width/height*256)) if height <= width else (math.floor(height/width*256), 256)
    tensor = torch.nn.functional.interpolate(tensor, size=size, mode='bilinear', align_corners=False)
    fast = f.center_crop(tensor, [256, 256])
    slow = fast.index_select(1, torch.linspace(0, 31, 8).long())
    return [slow, fast]
