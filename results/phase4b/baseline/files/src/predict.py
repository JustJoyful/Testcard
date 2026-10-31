"""Raw-video CPU inference for the frozen full-data ResNet + ridge model."""
import argparse
import json
import math
import sys
import warnings
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from src.baselines import predict_ridge
from src.cache import atomic_array
from src.features import extract_features, load_backbone
from src.preprocess import iter_samples, probe_video, spatial_tensor
from src.serving_bundle import WEIGHTS_SHA256, verify_bundle, verify_file


class PredictionError(ValueError):
    def __init__(self, message, status_code=422):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class InferenceRuntime:
    model: torch.nn.Module
    artifact: dict
    config: dict
    verified: dict
    startup_seconds: float
    limits: dict


def load_runtime(bundle: Path) -> InferenceRuntime:
    start = perf_counter()
    checked = verify_bundle(bundle)
    torch.set_num_threads(4)
    model = load_backbone('resnet50', 'cpu', checked['config'])
    verify_file(checked['weights'], WEIGHTS_SHA256)
    return InferenceRuntime(model, checked['artifact'], checked['config'], checked['verified'],
                            perf_counter() - start, checked['manifest']['limits'])


def validate_media(video, limits):
    video = Path(video)
    if not video.is_file():
        raise PredictionError('Provide a local MP4, MOV or WebM file.')
    size = video.stat().st_size
    if size > limits['file_bytes']:
        raise PredictionError('Video exceeds the 100 MiB file limit.', 413)
    if size == 0:
        raise PredictionError('The video is empty.')
    with video.open('rb') as handle:
        header = handle.read(4096)
    mp4 = len(header) >= 12 and header[4:8] == b'ftyp' and header[8:12] in (
        b'isom', b'iso2', b'iso3', b'iso4', b'iso5', b'iso6', b'avc1', b'mp41', b'mp42',
        b'qt  ', b'M4V ', b'MSNV', b'dash')
    webm = header.startswith(b'\x1a\x45\xdf\xa3') and b'webm' in header
    if not (mp4 or webm):
        raise PredictionError('Unsupported container: use MP4, MOV or WebM.', 415)
    try:
        meta = probe_video(video, timestamps=False)
    except ValueError as exc:
        raise PredictionError('Cannot read video metadata; check the file is a valid video.') from exc
    keys = ('duration_s', 'fps', 'width', 'height', 'rotation_applied')
    if any(key not in meta or not math.isfinite(meta[key]) for key in keys):
        raise PredictionError('Video metadata is missing or nonfinite.')
    width, height = meta['width'], meta['height']
    if (width <= 0 or height <= 0 or meta['duration_s'] <= 0 or meta['fps'] <= 0 or
            meta['duration_s'] > limits['duration_seconds'] or meta['fps'] > limits['fps'] or
            width * height > limits['source_pixels'] or max(width, height) > limits['longest_side'] or
            meta['rotation_applied'] not in (0, 90, 180, 270)):
        raise PredictionError('Video must be at most 30 seconds, 1920 pixels per side, '
                              '2073600 pixels and 60 fps, with valid orientation.')
    return meta


def predict_video(video: Path, runtime: InferenceRuntime, *, features_out: Path | None = None) -> dict:
    start = perf_counter()
    try:
        if features_out is not None:
            output = Path(features_out)
            if (output.resolve() == Path(video).resolve() or
                    (output.exists() and Path(video).exists() and output.samefile(video))):
                raise PredictionError('Feature output must be a different file from the input video.')
        validate_media(video, runtime.limits)
        rows = []
        with closing(iter_samples(Path(video), runtime.config)) as samples:
            for _, frame, _ in samples:
                values = extract_features(runtime.model, spatial_tensor(frame).unsqueeze(0), 'resnet50', 'cpu')
                row = values['pooled'].cpu().numpy()
                if row.shape != (1, 2048) or row.dtype != np.float32 or not np.isfinite(row).all():
                    raise ValueError('invalid ResNet feature shape/precision')
                rows.append(row[0].copy())
        if not rows:
            raise ValueError('no decoded samples')
        features = np.stack(rows)
        extracted = perf_counter()
        pooled = features.mean(axis=0, dtype=np.float64)[None, :]
        score = float(predict_ridge(runtime.artifact, pooled)[0])
        if not math.isfinite(score):
            raise ValueError('nonfinite score')
        headed = perf_counter()
        if features_out is not None:
            atomic_array(features_out, features)
        return dict(score=score, timings_seconds=dict(extraction=extracted-start, head=headed-extracted,
                    total=perf_counter()-start), verified=dict(runtime.verified),
                    startup_seconds=runtime.startup_seconds)
    except PredictionError:
        raise
    except (ValueError, OSError, RuntimeError) as exc:
        raise PredictionError('Video decoding or feature extraction failed; check the media file.') from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--bundle', type=Path, default=Path('artifacts/serving/manifest.json'))
    parser.add_argument('--features-out', type=Path)
    parser.add_argument('--json', action='store_true', help='Print full JSON instead of the terminal summary.')
    args = parser.parse_args(argv)
    try:
        runtime = load_runtime(args.bundle)
        with warnings.catch_warnings():
            # The frozen transform converts uint8 frames to a new float tensor before writing.
            warnings.filterwarnings('ignore', message=r'The given NumPy array is not writable, and PyTorch does not support non-writable tensors.*',
                                    category=UserWarning, module=r'src\.preprocess')
            result = predict_video(args.video, runtime, features_out=args.features_out)
        if args.json:
            print(json.dumps(result, allow_nan=False))
        else:
            duration = probe_video(args.video, timestamps=False)['duration_s']
            print(f'Video: {args.video.name}')
            print(f"Predicted quality: {result['score']:.2f} (KoNViD 1–5 scale)")
            print('Higher is better. Predictions are unclipped.\n')
            print(f"Analysis time: {result['timings_seconds']['total']:.2f} s for a {duration:.2f} s clip")
            print(f"Feature extraction (incl. backbone): {result['timings_seconds']['extraction']:.3f} s")
            print(f"Ridge head: {result['timings_seconds']['head']:.6f} s")
            print(f"Model startup: {result['startup_seconds']:.2f} s")
            print(f"CPU threads: {result['verified']['cpu_threads']}")
            print(f"Model {result['verified']['model_sha256'][:8]}… · Weights {result['verified']['weights_sha256'][:8]}… (verified)")
            print(f"Preprocessing SHA256: {result['verified']['preprocessing_sha256']}")
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f'Prediction failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
