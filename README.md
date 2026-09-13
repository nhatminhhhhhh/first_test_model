
### Windows setup

From the repository folder in PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run the commands with the interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
### Linux setup

From the repository folder in a terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `venv` is not installed on Ubuntu/Debian:

```bash
sudo apt update
sudo apt install python3-venv
```

### Windows Run detection

Detect an image:

```powershell
python detect.py --source path\to\image.jpg --conf 0.4
```

Detect a video:

```powershell
python detect.py --source path\to\video.mp4 --conf 0.4
```

Use the default webcam:

```powershell
python detect.py --source 0 --conf 0.4
```

Results are written to `runs\detect\predict`. Use `--model path\to\model.pt` if the
weights file is stored elsewhere. The model contains the classes `drowning`,
`out of water`, and `swimming`.

### Linux Run detection

Detect an image:

```bash
python detect.py --source path/to/image.jpg 
```

Detect a video:

```bash
python detect.py --source path/to/video.mp4 
```

Use the default webcam and display the annotated frames:

```bash
python detect.py --source 0 \
  --camera-width 640 --camera-height 480 --camera-fps 30
```

Press `q` in the detection window to stop the camera. Results are written to
`runs/`. If the Linux webcam is not device `0`, try another index
such as `1`, or pass a device path such as `/dev/video0`.

On a headless Linux server without a desktop, camera display with OpenCV may not
be available. In that case, use an image or video source a.

### Raspberry Pi 4 with OV5647

Export the PyTorch model to NCNN on a computer with the project environment:

```powershell
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\yolo.exe export model=model.pt format=ncnn imgsz=416
```

Copy the generated `model_ncnn_model` directory to the Raspberry Pi. Install the
camera and Python runtime packages on the Pi:

```bash
sudo apt update
sudo apt install -y python3-picamera2
python -m pip install -r requirements.txt
```

Run the OV5647-only application:

```bash
python main.py --model model_ncnn_model --imgsz 416 --width 640 --height 480
```

Press `q` to stop. If the frame rate is low, try `--imgsz 320`. Use
`--no-display` when running without a desktop.