# 🎯 Face Recognition System - Hệ Thống Nhận Diện Khuôn Mặt

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-1.15-orange.svg)](https://tensorflow.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org)

Hệ thống nhận diện khuôn mặt real-time sử dụng **FaceNet** và **MTCNN** với khả năng thu thập dữ liệu, huấn luyện và nhận diện trực tiếp từ camera.

## 📋 Mục Lục

- [🚀 Cài Đặt](#-cài-đặt)
- [📊 Thu Thập Dữ Liệu](#-thu-thập-dữ-liệu)
- [🎯 Huấn Luyện Model](#-huấn-luyện-model)
- [🎥 Chạy Nhận Diện](#-chạy-nhận-diện)
- [📂 Cấu Trúc Thư Mục](#-cấu-trúc-thư-mục)
- [⚙️ Cấu Hình](#️-cấu-hình)
- [🔧 Troubleshooting](#-troubleshooting)

## 🚀 Cài Đặt

### 1. Clone Repository

```bash
git clone <repository-url>
cd FACEID
```

### 2. Tạo Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Pre-trained Models

**FaceNet Model:**
- Download `20180402-114759.pb` từ [FaceNet Models](https://github.com/davidsandberg/facenet#pre-trained-models)
- Đặt vào thư mục `Models/`

**MTCNN Weights:**
- Download `det1.npy`, `det2.npy`, `det3.npy`
- Đặt vào thư mục `src/align/`

## 📊 Thu Thập Dữ Liệu

### Bước 1: Chạy Script Thu Thập

```bash
cd src
python collect_data.py
```

### Bước 2: Hướng Dẫn Sử Dụng

1. **Nhập tên người**: Khi chương trình khởi động, nhập tên của người cần thu thập dữ liệu
2. **Vị trí camera**: Đặt khuôn mặt trong khung hình xanh
3. **Thu thập**: 
   - Chương trình tự động chụp khi phát hiện khuôn mặt
   - Thu thập **30-50 ảnh** cho mỗi người
   - Thay đổi góc độ, ánh sáng, biểu cảm

### Bước 3: Kiểm Tra Dữ Liệu

```bash
# Kiểm tra số lượng ảnh đã thu thập
ls DataSet/FaceData/raw/[TÊN_NGƯỜI]/
```

### ⚠️ Lưu Ý Quan Trọng

- **Chất lượng ảnh**: Đảm bảo ảnh rõ nét, ánh sáng tốt
- **Đa dạng góc độ**: Thu thập ở nhiều góc độ khác nhau
- **Số lượng**: Tối thiểu 30 ảnh, tối ưu 50+ ảnh
- **Một người/lần**: Chỉ thu thập một người trong một lần chạy

## 🎯 Huấn Luyện Model

### Bước 1: Tiền Xử Lý Dữ Liệu

```bash
cd src
python align_dataset_mtcnn.py \
    --input_dir ../DataSet/FaceData/raw \
    --output_dir ../DataSet/FaceData/processed \
    --image_size 182 \
    --margin 44
```

### Bước 2: Huấn Luyện Classifier

```bash
python classifier.py \
    --mode TRAIN \
    --data_dir ../DataSet/FaceData/processed \
    --model ../Models/20180402-114759.pb \
    --classifier_filename ../Models/improved_facemodel.pkl
```

### Bước 3: Cải Thiện Model (Tùy Chọn)

```bash
python improve_recognition.py
```

## 🎥 Chạy Nhận Diện

### Nhận Diện Qua Camera

```bash
cd src
python face_rec_cam.py
```

### Nhận Diện Qua Video

```bash
python face_rec_cam.py --path path/to/video.mp4
```

### ⌨️ Phím Tắt

- **`q`**: Thoát chương trình
- **`ESC`**: Dừng thu thập dữ liệu

## 📂 Cấu Trúc Thư Mục

```
FACEID/
├── 📁 DataSet/
│   └── 📁 FaceData/
│       ├── 📁 raw/              # Ảnh gốc theo tên người
│       │   ├── 📁 Nguyen_Van_A/
│       │   ├── 📁 Tran_Thi_B/
│       │   └── 📁 ...
│       └── 📁 processed/        # Ảnh đã tiền xử lý
│           ├── 📁 Nguyen_Van_A/
│           └── 📁 ...
├── 📁 Models/
│   ├── 📄 20180402-114759.pb           # FaceNet model
│   ├── 📄 facemodel.pkl                # Classifier cũ
│   ├── 📄 improved_facemodel.pkl       # Classifier cải thiện
│   └── 📄 model-20180402-114759.*      # Checkpoint files
├── 📁 src/
│   ├── 📄 collect_data.py              # Thu thập dữ liệu
│   ├── 📄 face_rec_cam.py              # Nhận diện real-time
│   ├── 📄 classifier.py                # Huấn luyện classifier
│   ├── 📄 improve_recognition.py       # Cải thiện model
│   ├── 📄 align_dataset_mtcnn.py       # Tiền xử lý
│   └── 📁 align/                       # MTCNN weights
│       ├── 📄 det1.npy
│       ├── 📄 det2.npy
│       └── 📄 det3.npy
├── 📄 requirements.txt
├── 📄 README.md
├── 📄 .gitignore
└── 📄 .gitattributes
```

## ⚙️ Cấu Hình

### Thông Số Quan Trọng

**`face_rec_cam.py`:**
```python
CONFIDENCE_THRESHOLD = 0.7    # Ngưỡng tin cậy (0.0-1.0)
MIN_FACE_SIZE = 40           # Kích thước mặt tối thiểu (pixels)
```

**Camera Settings:**
```python
MINSIZE = 20                 # Kích thước mặt tối thiểu cho MTCNN
THRESHOLD = [0.6, 0.7, 0.7]  # Ngưỡng MTCNN
FACTOR = 0.709               # Scale factor
```

### Tùy Chỉnh Hiệu Suất

**GPU Memory:**
```python
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
```

**Input Size:**
```python
INPUT_IMAGE_SIZE = 160       # Kích thước đầu vào FaceNet
```

## 🔧 Troubleshooting

### ❌ Lỗi Thường Gặp

**1. "No module named 'tensorflow'"**
```bash
pip install tensorflow==1.15.0
# Hoặc cho GPU:
pip install tensorflow-gpu==1.15.0
```

**2. "Cannot find MTCNN weights"**
```bash
# Download và đặt file .npy vào src/align/
wget https://github.com/davidsandberg/facenet/raw/master/src/align/det1.npy
```

**3. "Camera not found"**
```bash
# Kiểm tra camera
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

**4. "Low recognition accuracy"**
- Thu thập thêm dữ liệu (50+ ảnh/người)
- Cải thiện chất lượng ảnh
- Điều chỉnh `CONFIDENCE_THRESHOLD`
- Chạy `improve_recognition.py`

### 🔍 Debug Mode

Để debug chi tiết:
```python
# Trong face_rec_cam.py, thêm:
print(f"🎯 Dự đoán: {best_name} (confidence: {confidence:.4f})")
```

### 📊 Đánh Giá Model

```bash
python classifier.py \
    --mode CLASSIFY \
    --data_dir ../DataSet/FaceData/processed \
    --model ../Models/20180402-114759.pb \
    --classifier_filename ../Models/improved_facemodel.pkl
```

## 🎯 Best Practices

### Thu Thập Dữ Liệu
1. **Ánh sáng tự nhiên** tốt hơn ánh sáng nhân tạo
2. **Nhiều góc độ**: thẳng, nghiêng trái/phải, nhìn lên/xuống
3. **Biểu cảm đa dạng**: cười, nghiêm túc, ngạc nhiên
4. **Phụ kiện**: với/không kính, mũ

### Huấn Luyện
1. **Cân bằng dữ liệu**: Số lượng ảnh tương đương cho mỗi người
2. **Chất lượng > Số lượng**: 30 ảnh tốt > 100 ảnh kém
3. **Regular retraining**: Huấn luyện lại khi thêm người mới

### Triển Khai
1. **Điều chỉnh threshold** theo môi trường thực tế
2. **Monitor performance** và log kết quả
3. **Backup models** trước khi update

## 📞 Hỗ Trợ

- **Issues**: Tạo issue trên GitHub
- **Documentation**: Xem comments trong code
- **Performance**: Kiểm tra system requirements

---

## 📝 Changelog

- **v1.0**: Basic face recognition
- **v1.1**: Improved data collection
- **v1.2**: Enhanced accuracy with better preprocessing
- **v1.3**: Real-time optimization

## 📜 License

MIT License - Xem file LICENSE để biết thêm chi tiết.

---

**�� Happy Coding!** 🎯 
