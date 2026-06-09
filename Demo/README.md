# ⛩️ MangaOCR — Neural Processing Pipeline

Hệ thống nhận diện chữ trên ảnh Manga sử dụng pipeline AI đa mô hình (SVTR, CRNN, TrOCR), kết hợp sức mạnh GPU từ Kaggle làm backend xử lý và giao diện React hiện đại làm frontend. Toàn bộ ứng dụng frontend được đóng gói bằng Docker để triển khai nhanh chóng trên mọi môi trường.

---

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────┐         Gradio Public URL          ┌──────────────────────┐
│   React Frontend    │ ◄──────────────────────────────────►│   Kaggle GPU Backend │
│   (Docker + Nginx)  │         @gradio/client              │   (demo-cs231.ipynb) │
│   localhost:8080    │                                     │   SVTR / CRNN / TrOCR│
└─────────────────────┘                                     └──────────────────────┘
```

- **Frontend** — Ứng dụng React/Vite được biên dịch thành file tĩnh, phục vụ qua Nginx trong Docker container.
- **Backend** — Notebook Python chạy trên Kaggle GPU, khởi tạo Gradio server và công khai API qua Public URL.

---

## 🚀 Hướng dẫn triển khai

### Yêu cầu

| Thành phần | Mô tả |
|---|---|
| [Docker Desktop](https://www.docker.com/products/docker-desktop/) | Đã cài đặt và đang chạy (biểu tượng trạng thái xanh lá). |
| Tài khoản [Kaggle](https://www.kaggle.com/) | Có quyền sử dụng GPU để chạy notebook backend. |

### Bước 1 — Khởi động Backend trên Kaggle

1. Mở notebook `demo-cs231.ipynb` trên Kaggle và chạy toàn bộ các cell.
2. Cell cuối cùng sẽ khởi tạo Gradio server và sinh ra một đường dẫn công khai:
   ```
   Running on public URL: https://xxxxxxxxxxxxxxxx.gradio.live
   ```
3. Sao chép đường dẫn này — đây chính là API endpoint mà frontend sẽ kết nối tới.

### Bước 2 — Build và chạy Frontend bằng Docker

1. Mở terminal tại thư mục dự án và build Docker image, truyền Gradio URL vừa copy vào:
   ```bash
   docker build --build-arg VITE_GRADIO_URL="https://xxxxxxxxxxxxxxxx.gradio.live" -t manga-ocr-frontend .
   ```
2. Khởi chạy container:
   ```bash
   docker run -d -p 8080:80 manga-ocr-frontend
   ```
3. Truy cập ứng dụng tại **http://localhost:8080**.

> **Lưu ý:** Mỗi khi Kaggle sinh ra Gradio URL mới, bạn cần build lại image với URL mới tương ứng.

---

## 🛠 Pipeline xử lý

Quy trình nhận diện chữ được chia thành 4 giai đoạn tuần tự:

| Giai đoạn | Tên | Mô tả |
|:-:|---|---|
| 1 | **Content Ingestion** | Tải ảnh Manga lên hệ thống. |
| 2 | **Neural Geometry Mapping** | Phát hiện và xác định tọa độ các vùng chứa chữ (Bounding Boxes). |
| 3 | **Snippet Isolation** | Cắt từng vùng chữ thành ảnh nhỏ riêng biệt để tăng độ chính xác. |
| 4 | **Recognition Manifest** | Nhận diện ký tự bằng model OCR đã chọn (SVTR, CRNN hoặc TrOCR). |

---

## 📦 Công nghệ sử dụng

| Lớp | Công nghệ |
|---|---|
| Frontend | React, Vite, TypeScript, TailwindCSS |
| UI Components | Lucide React, Framer Motion |
| Backend Communication | `@gradio/client` |
| AI Engine | Python, OpenCV, PaddlePaddle, PyTorch, Transformers |
| Containerization | Docker, Nginx |

---

<div align="center">
  <sub>Built for CS231 — Advanced Computer Vision</sub>
</div>
