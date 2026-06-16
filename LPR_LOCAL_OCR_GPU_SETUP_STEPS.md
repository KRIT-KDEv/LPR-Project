# LPR Local OCR GPU Setup & YOLO Integration Notes

## Goal

Set up the ANPR/LPR pipeline locally so it no longer depends on the remote OCR API.

Final target flow:

```text
YOLO local model
↓
Detect / track license plate
↓
Crop plate image
↓
Send crop to local OCR server
↓
OCR returns plate number / province / confidence
↓
YOLO script draws bounding box + OCR result on output video
```

---

## Current Local Folder Structure

Current project structure:

```text
LPR_LOCAL/
├── models/
│   └── best_v2.pt
│
├── ocr-server/
│   ├── __pycache__/
│   ├── .gitkeep
│   ├── Dockerfile
│   ├── main.py
│   ├── requirements.txt
│   └── venv/
│
├── ocr1_1/
│   ├── added_tokens.json
│   ├── chat_template.jinja
│   ├── config.json
│   ├── generation_config.json
│   ├── merges.txt
│   ├── model.safetensors
│   ├── preprocessor_config.json
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   ├── video_preprocessor_config.json
│   └── vocab.json
│
├── senior_plate_yolo/
│   ├── images/
│   └── labels/
│
├── test_images/
├── test_videos/
└── lpr_yolo_ocr_api_video.py
```

Notes:

- `models/best_v2.pt` is the YOLO license plate detection model.
- `ocr1_1/` is the OCR model folder. It must include the full model folder, not just one file.
- `ocr-server/main.py` is the FastAPI OCR server.
- `senior_plate_yolo/` is a YOLO dataset folder. It is not needed for OCR inference unless retraining YOLO.

---

## Important Concept Clarification

Local mode does **not** mean no server.

There are two local components:

```text
ocr-server/main.py
= local OCR API server
= receives cropped plate image
= returns OCR result

lpr_yolo_ocr_api_video.py
= YOLO pipeline script
= opens video
= runs YOLO detect/track
= crops plates
= sends crop to local OCR server
= saves output video and terminal summary
```

So the local setup normally uses 2 terminals:

```text
Terminal 1: Run OCR server
Terminal 2: Run YOLO + OCR video pipeline
```

---

## Step 1 — Rename OCR Weight File

The OCR model weight file originally came as:

```text
model-001.safetensors
```

It should be renamed to:

```text
model.safetensors
```

Final expected path:

```text
LPR_LOCAL/ocr1_1/model.safetensors
```

Reason:

Most Hugging Face / Transformers model loaders expect the main weight file to be named `model.safetensors` unless an index file tells it otherwise.

---

## Step 2 — Fix OCR Model Path in `main.py`

Original server path pointed to Jetson/server path:

```python
MODEL_PATH = os.getenv(
    "OCR_MODEL_PATH",
    "/mnt/ssd/ANPR/MODELstore/OCR/ocr1_1",
)
```

This was changed to a local relative path:

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = os.getenv(
    "OCR_MODEL_PATH",
    str((BASE_DIR / "../ocr1_1").resolve()),
)
```

Reason:

`main.py` runs from:

```text
LPR_LOCAL/ocr-server/main.py
```

The OCR model folder is located at:

```text
LPR_LOCAL/ocr1_1/
```

So from `ocr-server`, the correct relative path is:

```text
../ocr1_1
```

---

## Step 3 — Add Upload Endpoint to `main.py`

Original `main.py` only supported:

```text
GET  /health
POST /ocr
```

But `POST /ocr` accepted base64 JSON only:

```json
{
  "image_b64": "..."
}
```

For YOLO integration, the script sends image crops as multipart files:

```python
requests.post(url, files={"file": f})
```

So `/ocr/upload` was added.

Required import update:

```python
from fastapi import FastAPI, HTTPException, UploadFile, File
```

Added helper:

```python
async def upload_to_pil(file: UploadFile) -> Image.Image:
    raw = await file.read()
    return Image.open(io.BytesIO(raw)).convert("RGB")
```

Added route:

```python
@app.post("/ocr/upload")
async def ocr_upload(file: UploadFile = File(...)):
    if model is None or processor is None:
        raise HTTPException(503, "Model not loaded")

    try:
        image = await upload_to_pil(file)
    except Exception as e:
        raise HTTPException(400, f"Cannot read uploaded image: {e}")

    result = run_ocr_pipeline(image)

    return JSONResponse(result)
```

After this, the OCR server supports:

```text
GET  /health
POST /ocr
POST /ocr/upload
```

---

## Step 4 — Windows Blocked `uvicorn.exe`

Running this failed:

```powershell
uvicorn main:app --host 127.0.0.1 --port 8002
```

Error:

```text
An Application Control policy has blocked this file
```

Cause:

Windows blocked direct execution of:

```text
venv/Scripts/uvicorn.exe
```

Fix:

Run Uvicorn as a Python module instead:

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8002
```

This avoids directly running `uvicorn.exe`.

---

## Step 5 — Missing `torchvision`

Initial server run failed with:

```text
ModuleNotFoundError: No module named 'torchvision'
```

Cause:

`qwen_vl_utils` imports `torchvision` internally.

Fix was to install a compatible `torchvision`, but this led to version mismatch first.

---

## Step 6 — Torch / Torchvision Version Mismatch

After installing `torchvision`, import failed with:

```text
RuntimeError: operator torchvision::nms does not exist
```

Cause:

`torch` and `torchvision` were installed from incompatible versions or CUDA builds.

The broken state included mismatched packages such as:

```text
torch 2.12.0+cu...
torchvision incompatible / wrong build
```

Fix:

Clean uninstall and reinstall pinned versions from the same CUDA build.

---

## Step 7 — Correct PyTorch GPU Stack Installation

In the `ocr-server` venv:

```powershell
cd C:\Users\advlb\Desktop\LPR_LOCAL\ocr-server
venv\Scripts\activate
```

Uninstall old torch stack:

```powershell
python -m pip uninstall torch torchvision torchaudio -y
python -m pip cache purge
```

Upgrade packaging tools:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

Install required base dependencies first:

```powershell
python -m pip install typing_extensions jinja2 networkx filelock fsspec sympy
```

Install pinned CUDA versions:

```powershell
python -m pip install --no-cache-dir --force-reinstall `
  "torch==2.12.0+cu126" `
  "torchvision==0.27.0+cu126" `
  --extra-index-url https://download.pytorch.org/whl/cu126
```

Important note:

`torchaudio` is not required for this OCR server, so it was not installed to avoid extra dependency conflicts.

---

## Step 8 — Transformers Version Issue

When Transformers was downgraded too far, server failed with:

```text
ValueError: The checkpoint you are trying to load has model type `qwen3_vl`
but Transformers does not recognize this architecture.
```

Cause:

The OCR model uses `qwen3_vl`, which needs a newer Transformers version.

Fix:

Install a Transformers version/source that supports Qwen3-VL.

The working check after fix was:

```powershell
python -c "from transformers import AutoProcessor, AutoModelForImageTextToText; print('transformers ok')"
```

Output:

```text
transformers ok
```

---

## Step 9 — Final Dependency Check Passed

Final checks run in `ocr-server` venv:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Output:

```text
2.12.0+cu126 12.6 True
```

Check torchvision:

```powershell
python -c "import torchvision; print(torchvision.__version__)"
```

Output:

```text
0.27.0+cu126
```

Check Qwen utils:

```powershell
python -c "from qwen_vl_utils import process_vision_info; print('qwen ok')"
```

Output:

```text
qwen ok
```

Check Transformers:

```powershell
python -c "from transformers import AutoProcessor, AutoModelForImageTextToText; print('transformers ok')"
```

Output:

```text
transformers ok
```

---

## Step 10 — OCR Server Successfully Runs on GPU

Server run command:

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8002
```

Successful log:

```text
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:ocr-server:Loading OCR model from C:\Users\advlb\Desktop\LPR_Local\ocr1_1
Loading weights: 100%
INFO:ocr-server:OCR model ready on cuda
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8002
```

Current status:

```text
OCR local server = running
Device = cuda
Port = 8002
Endpoint = http://127.0.0.1:8002
```

Do not close this terminal while testing YOLO pipeline.

---

## Step 11 — Next Test: OCR Upload Endpoint

Before running YOLO video, test OCR local upload first.

Create this file in:

```text
LPR_LOCAL/test_ocr_local.py
```

Code:

```python
import requests

url = "http://127.0.0.1:8002/ocr/upload"
image_path = "test_images/test_car1.png"

with open(image_path, "rb") as f:
    response = requests.post(url, files={"file": f}, timeout=180)

print("Status:", response.status_code)

try:
    print(response.json())
except Exception:
    print(response.text)
```

Run in a second terminal:

```powershell
cd C:\Users\advlb\Desktop\LPR_LOCAL
python test_ocr_local.py
```

Expected result:

```text
Status: 200
```

With JSON like:

```json
{
  "plate_text": "...",
  "plate_number": "...",
  "plate_province": "...",
  "plate_color": null,
  "confidence": 0.92,
  "raw": "..."
}
```

---

## Step 12 — Connect to YOLO Local Pipeline

In:

```text
LPR_LOCAL/lpr_yolo_ocr_api_video.py
```

Use this config:

```python
YOLO_MODEL_PATH = "models/best_v2.pt"
VIDEO_PATH = "test_videos/test_car_video.mp4"
API_URL = "http://127.0.0.1:8002/ocr/upload"
```

Pipeline flow:

```text
best_v2.pt
↓
YOLO detect / track plate
↓
Crop plate image
↓
POST crop to http://127.0.0.1:8002/ocr/upload
↓
Receive OCR JSON
↓
Draw bounding box + plate result
↓
Save output video
↓
Print terminal summary
```

Run from second terminal:

```powershell
cd C:\Users\advlb\Desktop\LPR_LOCAL
python lpr_yolo_ocr_api_video.py
```

---

## Step 13 — If GPU VRAM Is Not Enough

OCR model is large. Running OCR server and YOLO on GPU at the same time may consume VRAM.

If CUDA out-of-memory occurs, reduce YOLO/OCR workload:

```python
IMG_SIZE = 640
OCR_EVERY_N_FRAMES = 60
MAX_OCR_PER_TRACK = 1
```

If stable, increase later:

```python
IMG_SIZE = 960
OCR_EVERY_N_FRAMES = 30
MAX_OCR_PER_TRACK = 2
```

---

## Key Lessons / Issues Fixed

### 1. Local server still needs to run

Local OCR means the server runs on your own machine:

```text
http://127.0.0.1:8002
```

It does not use the remote Cloudflare API anymore.

### 2. `main.py` and YOLO script do different jobs

```text
ocr-server/main.py
= OCR API service

lpr_yolo_ocr_api_video.py
= YOLO video pipeline
```

### 3. Qwen3-VL needs correct Transformers support

Older Transformers versions do not recognize:

```text
qwen3_vl
```

### 4. Torch and torchvision must match

Working stack:

```text
torch 2.12.0+cu126
torchvision 0.27.0+cu126
CUDA available: True
```

### 5. Run Uvicorn through Python module

Use:

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8002
```

Do not use:

```powershell
uvicorn main:app --host 127.0.0.1 --port 8002
```

because Windows Application Control may block `uvicorn.exe`.

---

## Current Final Status

```text
[OK] OCR model path fixed
[OK] model.safetensors prepared
[OK] /ocr/upload added
[OK] torch GPU installed
[OK] torchvision compatible
[OK] qwen_vl_utils import works
[OK] transformers import works
[OK] OCR server runs on CUDA
[Next] Test /ocr/upload with image
[Next] Run YOLO + OCR local video script
```
