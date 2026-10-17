import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # before torch is imported

import shutil, subprocess, tempfile, time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from auralis_core import load_auralis, predict

MODEL_DIR = Path(__file__).parent / "model"
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = time.time()
    state["model"], state["cfg"] = load_auralis(MODEL_DIR)   # loaded ONCE at startup
    print(f"Auralis loaded on {next(state['model'].parameters()).device} in {time.time() - t:.1f}s")
    yield
    state.clear()


app = FastAPI(title="Auralis API", version="0.1.0", lifespan=lifespan)

# lets the React dev server (Vite runs on 5173) call this API. Tightened on Day 13.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


def to_wav(src: Path, dst: Path) -> None:
    # any audio/video container -> 16 kHz mono wav (the format the model was trained on)
    cmd = ["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", str(dst)]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60)
    except FileNotFoundError:
        raise HTTPException(500, "ffmpeg is not installed on the server")
    except subprocess.TimeoutExpired:
        raise HTTPException(400, "File took too long to process")
    if r.returncode != 0 or not dst.exists():
        raise HTTPException(400, "Could not read this audio/video file")


@app.post("/predict")
def predict_emotion(file: UploadFile = File(...)):
    # a temp folder that is deleted automatically when the block ends (nothing is kept)
    with tempfile.TemporaryDirectory() as tmp:
        src, wav = Path(tmp) / "upload", Path(tmp) / "audio.wav"
        with open(src, "wb") as f:
            shutil.copyfileobj(file.file, f)
        to_wav(src, wav)
        t = time.time()
        result = predict(state["model"], state["cfg"], str(wav))
    result["inference_sec"] = round(time.time() - t, 3)
    return result