# Nhận diện đuối nước

Dự án gồm hai chương trình:

- `detect.py`: chương trình tổng quát , dùng để nhận diện
  hình ảnh, video, webcam và URL. Model mặc định là `model_ncnn_model`.
- `main.py`: chương trình chỉ sử dụng camera OV5647
  và bộ chạy model NCNN.

Các lớp nhận diện của model gồm `drowning`, `out of water` và `swimming`.

## Sử dụng `detect.py`

### Các tham số dòng lệnh

| Tham số | Giá trị mặc định | Mô tả |
| --- | --- | --- |
| `--source` | bắt buộc | Đường dẫn ảnh, video, URL hoặc số hiệu camera, ví dụ `0` |
| `--model` | `model_ncnn_model` | Thư mục model NCNN hoặc đường dẫn đến model `.pt` |
| `--conf` | `0.2` | Ngưỡng tin cậy nhận diện, từ `0` đến `1` |
| `--imgsz` | `416` | Kích thước ảnh dùng để suy luận |
| `--output` | `runs/` | Thư mục lưu kết quả |
| `--camera-width` | `640` | Chiều rộng webcam yêu cầu |
| `--camera-height` | `480` | Chiều cao webcam yêu cầu |
| `--camera-fps` | `30` | Số FPS webcam yêu cầu |

Các đường dẫn tương đối được tính từ thư mục dự án. Nhấn `q` để dừng khi
chạy chế độ camera.

### Nhận diện một hình ảnh

```bash
python detect.py --source duong/dan/toi/image.jpg
```

Ví dụ sử dụng ngưỡng tin cậy `0.4`:

```bash
python detect.py --source duong/dan/toi/image.jpg --conf 0.4
```

### Nhận diện video

```bash
python detect.py --source duong/dan/toi/video.mp4 --conf 0.4
```

Khi xử lý video từ một file cục bộ, chương trình sẽ hiển thị tiến trình từ
`0%` đến `100%` ngay trên cửa sổ dòng lệnh. Tiến trình dựa trên tổng số
khung hình mà OpenCV đọc được từ video. Với URL hoặc nguồn phát trực tiếp
không cung cấp tổng số khung hình, chương trình vẫn xử lý bình thường nhưng
không hiển thị phần trăm hoàn thành.

### Nhận diện bằng webcam

```bash
python detect.py \
  --source 0 \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 15 \
  --imgsz 320
```

Model mặc định là thư mục NCNN. Có thể chỉ định model khác bằng `--model`:

```bash
python detect.py --model model_ncnn_model --source 0
python detect.py --model model.pt --source duong/dan/toi/video.mp4
```

Kết quả ảnh và video được lưu trong thư mục chỉ định bởi `--output`. Khi chạy
với camera, video được lưu thành `camera.avi`.

## Sử dụng `main.py` cho camera OV5647

`main.py` dành cho Raspberry Pi 4 kết nối camera OV5647 qua giao tiếp CSI.
Chương trình sử dụng `Picamera2`, nhận khung hình BGR, chạy model NCNN, vẽ
kết quả nhận diện và giữ nguyên thuật toán xác nhận đuối nước trong hai giây.

### Các tham số dòng lệnh

| Tham số | Giá trị mặc định | Mô tả |
| --- | --- | --- |
| `--model` | `model_ncnn_model` | Thư mục model NCNN |
| `--conf` | `0.2` | Ngưỡng tin cậy nhận diện |
| `--iou` | `0.45` | Ngưỡng IoU khi loại bỏ hộp trùng lặp (NMS) |
| `--imgsz` | `0` | `0` dùng kích thước model trong `metadata.yaml` |
| `--width` | `640` | Chiều rộng khung hình OV5647 |
| `--height` | `480` | Chiều cao khung hình OV5647 |
| `--fps` | `15` | FPS camera và video đầu ra |
| `--threads` | `4` | Số luồng CPU dành cho NCNN |
| `--max-fps` | `10` | FPS tối đa; `0` để tắt giới hạn |
| `--thermal-limit` | `58` | Nhiệt độ SoC bắt đầu kích hoạt thời gian nghỉ bổ sung |
| `--output` | `runs/detect/ov5647.avi` | Đường dẫn video đầu ra |
| `--save` | tắt | Lưu video đã vẽ; có thể làm giảm hiệu năng |
| `--no-display` | tắt | Chạy không mở cửa sổ OpenCV |
| `--warmup` | `5` | Số khung hình khởi động trước khi đo |
| `--bench` | `0` | Xử lý N khung hình rồi thoát; `0` chạy liên tục |
| `--log-every` | `10` | In thông tin hiệu năng sau mỗi N khung hình |

### Chạy và hiển thị kết quả

```bash
python main.py \
  --model model_ncnn_model \
  --width 640 \
  --height 480 \
  --threads 4
```

Nhấn `q` để dừng chương trình.

### Chạy không giao diện

```bash
python main.py \
  --model model_ncnn_model \
  --width 640 \
  --height 480 \
  --no-display
```

Thêm `--save` nếu cần lưu video đã được vẽ kết quả:

```bash
python main.py \
  --no-display \
  --save \
  --output runs/detect/ov5647.avi
```

### Cấu hình tiết kiệm tài nguyên

```bash
python main.py \
  --imgsz 320 \
  --width 480 \
  --height 360 \
  --threads 2 \
  --max-fps 8 \
  --no-display
```

### Đo hiệu năng

Xử lý 100 khung hình, in thông tin hiệu năng rồi thoát:

```bash
python main.py \
  --bench 100 \
  --no-display \
  --threads 4 \
  --log-every 10
```
