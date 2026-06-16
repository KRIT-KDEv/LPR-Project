# Thai LPR Local Dashboard

Local Dashboard สำหรับตรวจจับป้ายทะเบียนไทยจากรูปภาพและวิดีโอ โดยใช้ YOLO สำหรับตรวจจับตำแหน่งป้ายทะเบียน และส่ง crop ป้ายไปยัง OCR Server แบบ Local เพื่ออ่านข้อความทะเบียน

โปรเจกต์นี้ออกแบบสำหรับรันบนเครื่องตัวเอง ไม่ใช่ production deployment

---

## Main Functions

ระบบนี้ทำงานหลัก ๆ ดังนี้:

```text
1. Upload Image / Video ผ่าน Streamlit Dashboard
2. ใช้ YOLO ตรวจจับตำแหน่งป้ายทะเบียน
3. Crop ภาพป้ายทะเบียนออกมา
4. ส่ง crop ไปยัง Local OCR Server
5. รับผล OCR กลับมา เช่น เลขทะเบียน / จังหวัด / confidence
6. แสดงผลบน Dashboard
7. สร้าง output image หรือ output video พร้อมกรอบและ label
8. บันทึกผลลัพธ์ลง outputs/
```

Feature ที่มีใน Dashboard:

```text
Upload Image
Upload Video
Preview input file
Analyze image/video
Show output video
Show province graph
Show timeline แบบ scroll
Show result table
Show OCR records
Show crop preview ที่ส่งเข้า OCR
Show analysis history
```

---

## Project Structure

โครงสร้างหลักของโปรเจกต์:

```text
LPR_LOCAL/
├── app.py
├── config.py
├── ocr_client.py
├── yolo_service.py
├── lpr_pipeline.py
├── result_utils.py
│
├── models/
│   └── best.pt
│
├── uploads/
│   ├── images/
│   └── videos/
│
├── outputs/
│
├── ocr-server/
│   ├── main.py
│   └── venv/
│
├── ocr1_1/
│   └── OCR model files
│
└── venv_app/
```

หมายเหตุ:

```text
ocr-server/  = OCR Server แยกต่างหาก
ocr1_1/      = OCR model folder
venv_app/    = Python environment ของ Dashboard
outputs/     = ผลลัพธ์ที่ระบบ generate
uploads/     = ไฟล์ที่ upload ผ่าน Dashboard
```

ถ้า push ขึ้น GitHub ควร ignore:

```text
ocr-server/
ocr1_1/
venv_app/
outputs/
uploads/
__pycache__/
.env
```

---

## Requirements

แนะนำให้ใช้ Python environment แยกเป็น 2 ชุด:

```text
1. venv_app
   ใช้สำหรับ Streamlit Dashboard + YOLO pipeline

2. ocr-server/venv
   ใช้สำหรับ OCR Server + OCR model
```

---

## Install Dashboard Environment

เข้า root project:

```powershell
cd /d C:\Users\advlb\Desktop\LPR_LOCAL
```

สร้าง environment:

```powershell
python -m venv venv_app
```

Activate:

```powershell
.\venv_app\Scripts\activate
```

ติดตั้ง package สำหรับ Dashboard:

```powershell
python -m pip install ultralytics opencv-python pillow numpy requests streamlit pandas plotly imageio-ffmpeg
```

Package หลักที่ใช้:

```text
streamlit       = Dashboard UI
ultralytics     = YOLO detection/tracking
opencv-python   = image/video processing
pillow          = draw Thai text on image/video
requests        = call OCR Server API
pandas          = result table
plotly          = graph
imageio-ffmpeg  = convert video to H.264 for browser preview
```

---

## Required Model File

YOLO model ต้องอยู่ที่:

```text
models/best.pt
```

ถ้ามี `best.pt` ตัวใหม่ ให้แทนที่ไฟล์เดิมที่ path นี้:

```text
LPR_LOCAL/models/best.pt
```

หลังเปลี่ยน model ให้ restart Streamlit ใหม่ เพราะ YOLO model ถูกโหลดค้างไว้ใน memory

---

## Config

ไฟล์ config หลักคือ:

```text
config.py
```

ค่าที่สำคัญ:

```python
YOLO_MODEL_PATH = BASE_DIR / "models" / "best.pt"

OCR_API_URL = "http://127.0.0.1:8002/ocr/upload"
OCR_HEALTH_URL = "http://127.0.0.1:8002/health"

CONF_THRES = 0.20
IMG_SIZE = 1280
MIN_DET_CONF_FOR_OCR = 0.25
```

ถ้า YOLO detect น้อยเกินไป สามารถลด threshold ได้ เช่น:

```python
CONF_THRES = 0.15
MIN_DET_CONF_FOR_OCR = 0.20
```

ข้อควรระวัง:

```text
ลด CONF_THRES = detect ได้มากขึ้น แต่ false positive อาจเพิ่ม
เพิ่ม IMG_SIZE = เห็นป้ายเล็กดีขึ้น แต่ประมวลผลช้าลง
```

---

## How to Run

ต้องเปิด 2 Terminal

---

### Terminal 1: Run OCR Server

เข้า folder OCR Server:

```powershell
cd /d C:\Users\advlb\Desktop\LPR_LOCAL\ocr-server
```

Activate OCR Server environment:

```powershell
.\venv\Scripts\activate
```

Run OCR Server:

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8002
```

รอจน server พร้อม เช่น:

```text
OCR model ready on cuda
Uvicorn running on http://127.0.0.1:8002
```

OCR Server endpoint ที่ Dashboard ใช้:

```text
http://127.0.0.1:8002/ocr/upload
```

---

### Terminal 2: Run Streamlit Dashboard

เข้า root project:

```powershell
cd /d C:\Users\advlb\Desktop\LPR_LOCAL
```

Activate Dashboard environment:

```powershell
.\venv_app\Scripts\activate
```

Run Dashboard:

```powershell
python -m streamlit run app.py
```

เปิดหน้าเว็บ:

```text
http://localhost:8501
```

---

## How to Use Dashboard

ลำดับใช้งาน:

```text
1. เปิด OCR Server
2. เปิด Streamlit Dashboard
3. กด Check OCR Server ที่ Sidebar
4. เลือก Image หรือ Video
5. Upload file
6. กด Analyze
7. รอระบบประมวลผล
8. ดูผลลัพธ์บน Dashboard
```

Dashboard จะแสดง:

```text
Output Video / Preview Image
Province Graph
Timeline
Result Table
OCR Details
OCR Crop Preview
Raw JSON
```

---

## Output Files

เมื่อ analyze เสร็จ ระบบจะสร้าง folder ใหม่ใน:

```text
outputs/
```

ตัวอย่าง:

```text
outputs/
└── video_20260610_125227/
    ├── result.json
    ├── output_raw.mp4
    ├── output_tracked_video.mp4
    ├── crops/
    └── uploaded_video.mp4
```

ความหมายไฟล์หลัก:

```text
result.json               = ผลลัพธ์ทั้งหมดของ analysis
output_tracked_video.mp4  = วิดีโอที่ใส่กรอบและ label แล้ว
output_raw.mp4            = วิดีโอก่อนแปลง H.264
crops/                    = รูป crop ป้ายที่ส่งเข้า OCR
uploaded_video.mp4        = ไฟล์ input ที่ upload เข้ามา
```

สำหรับ image:

```text
outputs/
└── image_20260610_120000/
    ├── result.json
    ├── preview_image.jpg
    ├── crops/
    └── uploaded_image.jpg
```

---

## Moving Project Folder

ถ้าย้ายโปรเจกต์จาก Desktop ไป Drive D เช่น:

```text
D:\LPR_LOCAL
```

คำสั่ง Terminal ต้องเปลี่ยนเป็น:

```powershell
cd /d D:\LPR_LOCAL
```

และ OCR Server:

```powershell
cd /d D:\LPR_LOCAL\ocr-server
```

ถ้าใช้ path แบบ relative ใน code ระบบควรยังทำงานได้ แต่ถ้ามี path แบบ hardcode ไปที่ `C:\Users\...` ต้องแก้ path ใหม่

---

## Common Issues

### 1. OCR Server not ready

ให้เช็กว่าเปิด OCR Server แล้วหรือยัง:

```powershell
cd /d C:\Users\advlb\Desktop\LPR_LOCAL\ocr-server
.\venv\Scripts\activate
python -m uvicorn main:app --host 127.0.0.1 --port 8002
```

จากนั้นกลับไปกด:

```text
Check OCR Server
```

---

### 2. Output Video เล่นไม่ได้ใน Streamlit

ระบบใช้ `imageio-ffmpeg` แปลงวิดีโอเป็น H.264

ติดตั้งด้วย:

```powershell
python -m pip install imageio-ffmpeg
```

ไฟล์ที่ควรใช้ดูผลคือ:

```text
output_tracked_video.mp4
```

---

### 3. เปลี่ยน Default Browser

Streamlit จะเปิดตาม default browser ของ Windows

เปลี่ยนได้ที่:

```text
Windows Settings
→ Apps
→ Default apps
→ เลือก Chrome / Brave / Edge
→ Set default
```

หรือ run แบบไม่เปิด browser อัตโนมัติ:

```powershell
python -m streamlit run app.py --server.headless true
```

แล้วเปิดเองที่:

```text
http://localhost:8501
```

---

## GitHub Notes

ถ้าจะ push ขึ้น GitHub แนะนำให้ `.gitignore` อย่างน้อยมี:

```gitignore
__pycache__/
*.pyc

venv/
venv_app/
.venv/
env/

ocr-server/
ocr1_1/

outputs/
uploads/

.env
.env.local
.env.*

.vscode/
.idea/
.DS_Store
Thumbs.db
```

ถ้า `models/best.pt` ใหญ่เกินไป ให้เพิ่ม:

```gitignore
models/*.pt
```

แล้ววาง model เองในเครื่องที่:

```text
models/best.pt
```

---

## Quick Start

Run OCR Server:

```powershell
cd /d C:\Users\advlb\Desktop\LPR_LOCAL\ocr-server
.\venv\Scripts\activate
python -m uvicorn main:app --host 127.0.0.1 --port 8002
```

Run Dashboard:

```powershell
cd /d C:\Users\advlb\Desktop\LPR_LOCAL
.\venv_app\Scripts\activate
python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

Use:

```text
Check OCR Server
Upload Image / Video
Analyze
View Result
```
