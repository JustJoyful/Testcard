import subprocess

import numpy as np
import pytest
import torch

from src import preprocess as p


def test_nearest_timestamp_tie_duplicates_and_boundaries():
    np.testing.assert_array_equal(p.nearest_indices(np.array([0., .1, .4]), np.array([.05, .3, -1, 2])), [0, 2, 0, 2])
    np.testing.assert_array_equal(p.nearest_indices(np.array([0., 0., .4]), np.array([0., .01])), [0, 0])
    with pytest.raises(ValueError):
        p.nearest_indices(np.array([]), np.array([0.]))
    with pytest.raises(ValueError):
        p.nearest_indices(np.array([1., 0.]), np.array([0.]))


@pytest.mark.parametrize('angle', [0, 90, 180, 270])
def test_rotation_matches_upright_pixel_positions(angle):
    frame = np.arange(18).reshape(2, 3, 3)
    np.testing.assert_array_equal(p.rotate_frame(frame, angle), np.rot90(frame, angle // 90))


def test_unsupported_rotation_fails():
    with pytest.raises(ValueError, match='rotation'):
        p.rotate_frame(np.zeros((2, 3, 3)), 45)


def test_spatial_normalization_and_aspect_preserving_resize():
    frame = np.full((40, 80, 3), [255, 0, 0], dtype=np.uint8)
    result = p.spatial_tensor(frame)
    assert result.shape == (3, 448, 448)
    torch.testing.assert_close(result[:, 200, 200], torch.tensor([(1-.485)/.229, -.456/.224, -.406/.225]))
    # A circle remains round when a wide frame is resized then cropped.
    yy, xx = np.mgrid[:40, :80]
    circle = np.repeat((((xx-40)**2+(yy-20)**2) < 8**2)[..., None], 3, axis=2).astype(np.uint8)*255
    mask = p.spatial_tensor(circle)[0] > 0
    ys, xs = mask.nonzero(as_tuple=True)
    assert abs((ys.max()-ys.min()).item() - (xs.max()-xs.min()).item()) <= 1


def test_motion_shape_normalization_and_reference_resize():
    frames = np.zeros((32, 24, 48, 3), dtype=np.uint8)
    slow, fast = p.motion_tensors(frames, {})
    assert slow.shape == (3, 8, 256, 256)
    assert fast.shape == (3, 32, 256, 256)
    torch.testing.assert_close(fast, torch.full_like(fast, -2.))
    # Reference packing selects linspace indices, not every fourth frame.
    frames[:] = np.arange(32)[:, None, None, None]
    slow, fast = p.motion_tensors(frames, {})
    torch.testing.assert_close(slow, fast[:, [0, 4, 8, 13, 17, 22, 26, 31]])


@pytest.fixture
def short_video(tmp_path):
    path = tmp_path / 'short.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=64x32:rate=10:duration=0.5', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(path)], check=True)
    return path


def test_sequential_decode_short_clip_boundary_replication(short_video):
    samples = list(p.iter_samples(short_video, {}))
    assert [t for t, _, _ in samples] == [0., .25]
    assert all(t < .5 for t, _, _ in samples)
    first, last = samples[0][2], samples[-1][2]
    assert first.shape == (32, 32, 64, 3)
    np.testing.assert_array_equal(first[0], first[14])
    np.testing.assert_array_equal(last[-1], last[-2])


def test_generated_container_rotation_is_applied(short_video, tmp_path):
    path = tmp_path / 'rotated.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-display_rotation', '90', '-i', str(short_video), '-c', 'copy', str(path)], check=True)
    assert p.probe_video(path)['rotation_applied'] == 90
    original = list(p.iter_samples(short_video, {}))[0][1]
    upright = list(p.iter_samples(path, {}))[0][1]
    np.testing.assert_array_equal(upright, np.rot90(original, 1))


def test_truncated_decode_is_failure(short_video, monkeypatch):
    real = p.probe_video
    def incorrect_count(path):
        meta = real(path)
        meta['pts'] = np.append(meta['pts'], .49)
        return meta
    monkeypatch.setattr(p, 'probe_video', incorrect_count)
    with pytest.raises(ValueError, match='short.mp4'):
        list(p.iter_samples(short_video, {}))


def test_inventory_metadata_probe_does_not_require_frame_timestamps(short_video):
    meta = p.probe_video(short_video, timestamps=False)
    assert meta['duration_s'] == .5
    assert meta['width'] == 64
    assert 'pts' not in meta


def test_slowfast_matches_installed_reference_on_synthetic_pixels():
    import importlib.util
    from pathlib import Path
    import pytorchvideo
    from torchvision.transforms import functional as f
    location = Path(pytorchvideo.__file__).parent/'transforms/functional.py'
    spec = importlib.util.spec_from_file_location('pytorchvideo_reference', location)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    pixels = np.random.default_rng(0).integers(0, 256, size=(32, 20, 40, 3), dtype=np.uint8)
    video = torch.from_numpy(pixels).permute(3, 0, 1, 2).float()/255
    expected = f.center_crop(reference.short_side_scale((video-.45)/.225, 256), [256, 256])
    slow, fast = p.motion_tensors(pixels, {})
    torch.testing.assert_close(fast, expected, rtol=0, atol=0)
    torch.testing.assert_close(slow, expected[:, [0, 4, 8, 13, 17, 22, 26, 31]], rtol=0, atol=0)


def test_metadata_only_probe_does_not_request_frame_sections(short_video, monkeypatch):
    original = p.subprocess.run
    def inspect_command(command, **kwargs):
        # ffprobe -show_entries frame=... implicitly enables full frame scanning.
        assert 'frame=' not in command[command.index('-show_entries')+1]
        return original(command, **kwargs)
    monkeypatch.setattr(p.subprocess, 'run', inspect_command)
    assert p.probe_video(short_video, timestamps=False)['duration_s'] == .5
