# auralis_core.py : audio utils, augmentation, model, inference for Auralis AI
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, sosfilt, fftconvolve
from transformers import Wav2Vec2Config, Wav2Vec2Model

SR = 16000
MAX_SEC = 4.0
MAX_LEN = int(SR * MAX_SEC)

# ------------------------------------------------------------------ audio utils
def load_wave(path, trim_db=30):
    # load as 16 kHz mono and trim leading/trailing silence
    import librosa
    x, _ = librosa.load(str(path), sr=SR, mono=True)
    if trim_db:
        y, _ = librosa.effects.trim(x, top_db=trim_db)
        if len(y) >= SR // 10:
            x = y
    return x.astype(np.float32)

def normalize(x):
    # per-utterance zero-mean / unit-variance (what Wav2Vec2 expects)
    x = x - x.mean()
    return (x / np.sqrt(x.var() + 1e-7)).astype(np.float32)

def crop_or_pad_start(x, max_len=MAX_LEN, rng=None):
    # random crop when training (rng given), center crop otherwise. Padding is done after normalization.
    n = len(x)
    if n > max_len:
        start = int(rng.integers(0, n - max_len + 1)) if rng is not None else (n - max_len) // 2
        x = x[start:start + max_len]
    return x

# ------------------------------------------------------------------ augmentations / corruptions
def _rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))

def colored_noise(n, kind, rng):
    white = rng.standard_normal(n).astype(np.float32)
    if kind == 'white':
        return white
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    power = {'pink': 1.0, 'brown': 2.0}[kind]
    spec = spec / (f ** (power / 2.0))
    return np.fft.irfft(spec, n).astype(np.float32)

def add_noise(x, snr_db, kind, rng):
    noise = colored_noise(len(x), kind, rng)
    noise = noise * (_rms(x) / (_rms(noise) * 10 ** (snr_db / 20.0)))
    return (x + noise).astype(np.float32)

def add_reverb(x, rt60, rng):
    # synthetic room impulse response: direct path + exponentially decaying noise tail
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n).astype(np.float32) * np.exp(-6.9 * t / rt60) * 0.5
    ir[0] = 1.0
    return fftconvolve(x, ir)[:len(x)].astype(np.float32)

def bandpass(x, lo, hi):
    sos = butter(6, [lo / (SR / 2), hi / (SR / 2)], btype='band', output='sos')
    return sosfilt(sos, x).astype(np.float32)

def lowpass(x, cutoff):
    sos = butter(6, cutoff / (SR / 2), btype='low', output='sos')
    return sosfilt(sos, x).astype(np.float32)

def clip_distort(x, factor):
    peak = float(np.max(np.abs(x))) + 1e-9
    return np.clip(x / peak * factor, -1.0, 1.0).astype(np.float32)

def change_speed(x, factor):
    n = int(len(x) / factor)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)

def augment_train(x, rng):
    # random chain of real-world degradations. Probabilities chosen so ~25% of samples stay clean.
    if rng.random() < 0.30:
        x = change_speed(x, rng.uniform(0.95, 1.05))
    if rng.random() < 0.30:
        x = add_reverb(x, rng.uniform(0.1, 0.8), rng)
    r = rng.random()
    if r < 0.10:
        x = bandpass(x, 300, 3400)                       # telephone
    elif r < 0.18:
        x = lowpass(x, rng.uniform(3500, 7000))          # cheap mic / low bandwidth
    if rng.random() < 0.65:
        x = add_noise(x, rng.uniform(3, 30), str(rng.choice(['white', 'pink', 'brown'])), rng)
    if rng.random() < 0.10:
        x = clip_distort(x, rng.uniform(2, 6))
    return x

def robustness_scenarios():
    # fixed test-time corruptions for the robustness benchmark: name -> fn(x, rng)
    return {
        'clean': None,
        'white noise 20 dB': lambda x, r: add_noise(x, 20, 'white', r),
        'white noise 10 dB': lambda x, r: add_noise(x, 10, 'white', r),
        'white noise 5 dB':  lambda x, r: add_noise(x, 5, 'white', r),
        'white noise 0 dB':  lambda x, r: add_noise(x, 0, 'white', r),
        'pink noise 10 dB':  lambda x, r: add_noise(x, 10, 'pink', r),
        'brown noise 10 dB': lambda x, r: add_noise(x, 10, 'brown', r),
        'reverb RT60 0.4 s': lambda x, r: add_reverb(x, 0.4, r),
        'reverb RT60 0.8 s': lambda x, r: add_reverb(x, 0.8, r),
        'telephone 300-3400 Hz': lambda x, r: bandpass(x, 300, 3400),
        'low-pass 4 kHz':    lambda x, r: lowpass(x, 4000),
        'clipping (x4)':     lambda x, r: clip_distort(x, 4.0),
        'speed 0.9x':        lambda x, r: change_speed(x, 0.9),
        'speed 1.1x':        lambda x, r: change_speed(x, 1.1),
        'noise 10 dB + reverb 0.4 s': lambda x, r: add_noise(add_reverb(x, 0.4, r), 10, 'pink', r),
    }

# ------------------------------------------------------------------ model
class AttentiveStatsPooling(nn.Module):
    # attention-weighted mean + std over time; also returns the attention weights (explainability)
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x, mask):
        x = x.float()
        a = self.att(x).squeeze(-1).masked_fill(~mask, -1e4)
        w = torch.softmax(a, dim=1).unsqueeze(-1)
        mean = (w * x).sum(1)
        var = (w * (x - mean.unsqueeze(1)) ** 2).sum(1)
        return torch.cat([mean, torch.sqrt(var + 1e-5)], dim=-1), w.squeeze(-1)

class AuralisModel(nn.Module):
    # Wav2Vec2 backbone -> learnable weighted sum of all layers -> projection -> attentive stats pooling -> classifier
    def __init__(self, backbone, num_labels, load_pretrained=True, proj_dim=256,
                 dropout=0.3, mask_time_prob=0.05):
        super().__init__()
        cfg = backbone if isinstance(backbone, Wav2Vec2Config) else Wav2Vec2Config.from_pretrained(backbone)
        cfg.mask_time_prob = mask_time_prob        # SpecAugment-style time masking (training only)
        cfg.mask_time_length = 10
        cfg.mask_feature_prob = 0.0
        cfg.layerdrop = 0.0
        cfg.hidden_dropout = 0.1
        cfg.attention_dropout = 0.1
        cfg.feat_proj_dropout = 0.05
        if load_pretrained and isinstance(backbone, str):
            self.backbone = Wav2Vec2Model.from_pretrained(backbone, config=cfg)
        else:
            self.backbone = Wav2Vec2Model(cfg)
        self.backbone.freeze_feature_encoder()     # CNN feature extractor stays frozen (standard practice)
        H = cfg.hidden_size
        self.layer_w = nn.Parameter(torch.zeros(cfg.num_hidden_layers + 1))
        self.proj = nn.Sequential(nn.Linear(H, proj_dim), nn.GELU(), nn.Dropout(dropout))
        self.pool = AttentiveStatsPooling(proj_dim)
        self.cls = nn.Sequential(nn.LayerNorm(proj_dim * 2), nn.Dropout(dropout),
                                 nn.Linear(proj_dim * 2, proj_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(proj_dim, num_labels))

    def forward(self, wave, lengths, return_attention=False):
        B, N = wave.shape
        amask = (torch.arange(N, device=wave.device)[None, :] < lengths[:, None]).long()
        use_mask = self.backbone.config.feat_extract_norm == 'layer'   # base models: no attention mask
        out = self.backbone(wave, attention_mask=amask if use_mask else None, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0)                      # L, B, T, H
        w = torch.softmax(self.layer_w, dim=0).to(hs.dtype)
        x = (w.view(-1, 1, 1, 1) * hs).sum(0)                           # B, T, H
        T = x.shape[1]
        flen = self.backbone._get_feat_extract_output_lengths(lengths).clamp(max=T)
        fmask = torch.arange(T, device=wave.device)[None, :] < flen[:, None]
        pooled, att = self.pool(self.proj(x), fmask)
        logits = self.cls(pooled)
        return (logits, att) if return_attention else logits

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():     # Apple Silicon (Metal)
        return 'mps'
    return 'cpu'

# ------------------------------------------------------------------ inference (used by the portal)
def load_auralis(root, device=None):
    root = Path(root)
    cfg = json.load(open(root / 'best_model' / 'auralis_config.json'))
    model = AuralisModel(cfg['backbone'], len(cfg['labels']), load_pretrained=False)
    ckpt = torch.load(root / 'best_model' / 'auralis_best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    device = device or get_device()
    return model.to(device).eval(), cfg

@torch.no_grad()
def predict(model, cfg, audio, stride_sec=2.0, batch=16):
    # audio: file path or 16 kHz float array. Long audio is split into overlapping 4 s windows;
    # returns the overall emotion (mean of window probabilities) plus a per-window timeline.
    device = next(model.parameters()).device
    x = load_wave(audio) if isinstance(audio, (str, Path)) else np.asarray(audio, dtype=np.float32)
    n = len(x)
    if n <= MAX_LEN:
        starts = [0]
    else:
        starts = list(range(0, n - MAX_LEN + 1, int(stride_sec * SR)))
        if starts[-1] + MAX_LEN < n:
            starts.append(n - MAX_LEN)
    segs, lens = [], []
    for s in starts:
        seg = normalize(x[s:s + MAX_LEN])
        pad = np.zeros(MAX_LEN, np.float32)
        pad[:len(seg)] = seg
        segs.append(pad); lens.append(len(seg))
    model.eval()
    probs = []
    for i in range(0, len(segs), batch):
        xb = torch.from_numpy(np.stack(segs[i:i + batch])).to(device)
        lb = torch.tensor(lens[i:i + batch]).to(device)
        probs.append(torch.softmax(model(xb, lb).float(), dim=-1).cpu().numpy())
    probs = np.concatenate(probs)
    mean = probs.mean(0)
    labels = cfg['labels']
    k = int(mean.argmax())
    return {
        'emotion': labels[k],
        'confidence': float(mean[k]),
        'probabilities': {l: float(p) for l, p in zip(labels, mean)},
        'duration_sec': n / SR,
        'timeline': [{'start': s / SR, 'end': min(s + MAX_LEN, n) / SR,
                      'emotion': labels[int(p.argmax())], 'confidence': float(p.max())}
                     for s, p in zip(starts, probs)],
    }
