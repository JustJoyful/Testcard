"""Frozen pretrained backbones shared by offline and serving inference."""
import hashlib
import re
from contextlib import nullcontext
from pathlib import Path

import torch


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_backbone(name: str, device: str, config: dict) -> torch.nn.Module:
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this execution environment')
    settings = config.get('backbones', {}).get('motion' if name in ('slowfast_r50', 'r3d18') else name, {})
    if name == 'dino' and not re.fullmatch('[0-9a-f]{40}', settings.get('hub_commit') or ''):
        raise ValueError('DINO requires a full pinned hub commit')
    hub = Path(config.get('weights_dir', 'weights')) / 'torch'
    hub.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(hub))
    try:
        if name == 'dino':
            model = torch.hub.load(f'facebookresearch/dinov2:{settings["hub_commit"]}',
                                   'dinov2_vitb14', pretrained=True, trust_repo=True,
                                   skip_validation=True)
            if getattr(model, 'num_register_tokens', 0):
                raise ValueError('DINO non-register architecture required')
            model._phase0_blocks = tuple(settings.get('blocks', [6, 9, 12]))
            checkpoint = hub / 'checkpoints/dinov2_vitb14_pretrain.pth'
        elif name == 'resnet50':
            from torchvision.models import ResNet50_Weights, resnet50
            weights = ResNet50_Weights.IMAGENET1K_V2
            model = resnet50(weights=weights)
            model.fc = torch.nn.Identity()
            checkpoint = hub / 'checkpoints' / Path(weights.url).name
        elif name == 'slowfast_r50':
            import pytorchvideo
            source = Path(pytorchvideo.__file__).parent / 'models/slowfast.py'
            if settings.get('source_sha256') and sha256(source) != settings['source_sha256']:
                raise ValueError('SlowFast source checksum mismatch')
            from pytorchvideo.models.hub import slowfast_r50
            model = slowfast_r50(pretrained=True)
            model.blocks[-1].proj = torch.nn.Identity()
            model.blocks[-1].activation = torch.nn.Identity()
            checkpoint = hub / 'checkpoints/SLOWFAST_8x8_R50.pyth'
        elif name == 'r3d18':
            from torchvision.models.video import R3D_18_Weights, r3d_18
            weights = R3D_18_Weights.KINETICS400_V1
            model = r3d_18(weights=weights)
            model.fc = torch.nn.Identity()
            checkpoint = hub / 'checkpoints' / Path(weights.url).name
        else:
            raise ValueError(f'unknown backbone {name}')
        expected = settings.get('weights_sha256')
        if expected and (not checkpoint.is_file() or sha256(checkpoint) != expected):
            raise ValueError(f'{name}: checkpoint checksum mismatch')
        model._phase0_checkpoint = checkpoint
        return model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
    except Exception as exc:
        raise RuntimeError(f'{name}: pretrained load failed: {exc}') from exc


@torch.inference_mode()
def extract_features(model, samples, name: str, device: str) -> dict[str, torch.Tensor]:
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    def move(tensor):
        return tensor.to(device=device, dtype=torch.float32)
    samples = [move(x) for x in samples] if isinstance(samples, list) else move(samples)
    autocast = torch.autocast('cuda', dtype=torch.float16) if device.startswith('cuda') else nullcontext()
    with autocast:
        if name == 'dino':
            blocks = getattr(model, '_phase0_blocks', (6, 9, 12))
            layers = model.get_intermediate_layers(samples, n=[n-1 for n in blocks], return_class_token=True, norm=True)
            result = {'cls': layers[-1][1]}
            for block, (patches, _) in zip(blocks, layers, strict=True):
                result[f'patchmean_L{block}'] = patches.mean(dim=1)
                result[f'patchstd_L{block}'] = patches.std(dim=1, correction=0)
        else:
            result = {'pooled': model(samples).flatten(1)}
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise ValueError(f'{name}: nonfinite features')
    return {key: value.detach().float() for key, value in result.items()}
