# Nhận Diện Chữ Cái Tiếng Nhật trong Truyện Tranh Manga

> **CS231.Q23 — Nhập Môn Thị Giác Máy Tính**  
> Trường Đại học Công Nghệ Thông Tin, ĐHQG TP.HCM  
> Giảng viên: **TS. Mai Tiến Dũng**

---

## Thành viên nhóm


| STT | Họ và tên       | MSSV     | Mức độ hoàn thành |
| --- | --------------- | -------- | ----------------- |
| 1   | Nguyễn Anh Quân | 23521259 | 100%              |
| 2   | Lê Đăng Khoa    | 23520740 | 100%              |
| 3   | Nguyễn Minh Đức | 23520312 | 100%              |


---

## Tổng quan

Đề tài tập trung vào bài toán **Text Recognition** chuyên biệt cho truyện tranh manga Nhật Bản — một miền ứng dụng đặc thù với nhiều thách thức:

- Ba bảng chữ cái xen kẽ: **Hiragana, Katakana, Kanji**
- Văn bản xuất hiện theo cả hai hướng **dọc và ngang**
- Sự xuất hiện của **furigana** (chú âm kích thước nhỏ bên cạnh Kanji)
- Phông chữ đa dạng, nền nhiễu, và văn bản chồng lấp với hình minh họa

Nhóm nghiên cứu, triển khai và đánh giá ba kiến trúc đại diện cho hai hướng tiếp cận chính trong nhận dạng văn bản: **CRNN** và **SVTR** (CTC-based), và **TrOCR** (Transformer/Attention-based).

---

## Dữ liệu

Nguồn dữ liệu: **[Manga109-s](http://www.manga109.org/en/)** — tập con công khai của Manga109, gồm ảnh trang manga kèm annotation XML.

Quy trình xử lý sinh ra **hai bộ dữ liệu song song**, mỗi bộ **60,000 mẫu** (48k train / 6k val / 6k test):


| Bộ dữ liệu       | Mô tả                                                               | Dùng cho   |
| ---------------- | ------------------------------------------------------------------- | ---------- |
| `bubble_dataset` | Ảnh crop nguyên gốc từ bong bóng thoại, giữ nguyên bố cục dọc/ngang | TrOCR      |
| `line_dataset`   | Ảnh đã chuyển đổi thành **dòng ngang duy nhất** (bubble → line)     | CRNN, SVTR |


Bước chuyển đổi **bubble → line** xử lý cả hai nhánh:

- **Ảnh dọc**: tách cột chữ, lọc furigana (< 70% chiều rộng cột lớn nhất), sắp xếp từ phải sang trái, xoay 90°, ghép ngang.
- **Ảnh ngang**: tách dòng chữ, lọc furigana (< 70% chiều cao dòng lớn nhất), ghép ngang.

Script xây dựng dataset: `[notebooks/final_data.ipynb](notebooks/final_data.ipynb)`

---

## Phương pháp

### CRNN — Convolutional Recurrent Neural Network

Biến thể sử dụng backbone **MobileNetV3-Small** (~2M tham số), warm-start từ pretrained tiếng Trung (`ch_ppocr_mobile_v2.0_rec_train`):

- **Backbone**: MobileNetV3-Small, scale = 0.5, input 3×32×320
- **Neck**: BiLSTM hai chiều, hidden size = 48
- **Head**: CTCHead ánh xạ ra từ điển tiếng Nhật **2,832 ký tự** 
- **Độ dài chuỗi tối đa**: 80 ký tự

### SVTR — Single Visual model for Scene Text Recognition

Biến thể **SVTR-Tiny** (~6M tham số), warm-start từ pretrained tiếng Trung (`rec_svtr_tiny_none_ctc_ch_train`):

- **Backbone**: SVTRNet, 3 stage, embedding [64, 128, 256], 12 Mixing Block (6 Local + 6 Global)
- **Neck**: reshape đơn thuần (không RNN), input 3×32×320
- **Head**: CTCHead ánh xạ ra từ điển tiếng Nhật **2,832 ký tự**

### TrOCR — Transformer-based OCR

Kiến trúc Encoder-Decoder thuần Transformer, không có CNN hay RNN:

- **Encoder**: ViT từ `microsoft/trocr-base-printed` (frozen phần lớn)
- **Decoder**: `cl-tohoku/bert-base-japanese-char-v2` — Japanese BERT với từ vựng 6,144 ký tự
- **LoRA**: fine-tune ~30M / 220M tham số (chỉ Cross-Attention + low-rank adapters), giảm 86% tham số cần cập nhật
- Huấn luyện 2 stage: Stage 1 (15 epoch) + Stage 2 (10 epoch)

---

## Huấn luyện

### CRNN & SVTR — RunPod (RTX 3090)

Cả hai mô hình PaddleOCR được huấn luyện trên nền tảng **[RunPod](https://www.runpod.io/)** với cấu hình:

- **GPU**: NVIDIA RTX 3090 · 24GB VRAM
- **CPU**: 32 vCPU · 125GB RAM
- **Framework**: PaddlePaddle GPU 3.0.0 (CUDA 12.6) + PaddleOCR
- **Workspace**: `/workspace`


| Siêu tham số  | CRNN                                        | SVTR                                        |
| ------------- | ------------------------------------------- | ------------------------------------------- |
| Loss          | CTCLoss                                     | CTCLoss                                     |
| Optimizer     | Adam + L2 (1e-5)                            | Adam + L2 (1e-5)                            |
| Learning rate | Cosine decay, max 1.5×10⁻³, warmup 8 ep     | Cosine decay, max 8×10⁻⁴, warmup 3 ep       |
| Batch size    | 512 / GPU                                   | 256 / GPU                                   |
| Epochs        | 100 (early stopping, patience 24×eval_step) | 100 (early stopping, patience 16×eval_step) |


Notebook huấn luyện:

- CRNN: `[notebooks/final_train_crnn.ipynb](notebooks/final_train_crnn.ipynb)`
- SVTR: `[notebooks/final_train_svtr.ipynb](notebooks/final_train_svtr.ipynb)`

### TrOCR — Kaggle (H100)

TrOCR được fine-tune trên **[Kaggle](https://www.kaggle.com/)** tận dụng GPU miễn phí với tối ưu cho H100:

- **Framework**: PyTorch + HuggingFace Transformers + PEFT (LoRA)
- **Optimizer**: AdamW (`adamw_torch_fused`), weight_decay = 0.01
- **Learning rate**: Cosine decay, max 10⁻⁴, warmup_ratio = 0.15
- **Effective batch size**: 48 (batch 12 × gradient accumulation 4)
- **TF32 + bf16** được bật để tận dụng kiến trúc H100

Notebook huấn luyện: `[notebooks/trocr-rec.ipynb](notebooks/trocr-rec.ipynb)`

---

## Kết quả

Đánh giá trên tập **test 6,000 mẫu** với ba độ đo: Exact Match Accuracy, CER (Character Error Rate), NED (Normalized Edit Distance).


| Mô hình   | Accuracy (%) ↑ | CER (%) ↓ | NED (%) ↓ |
| --------- | -------------- | --------- | --------- |
| CRNN      | 41.83          | 33.17     | 26.14     |
| SVTR      | 45.81          | 31.16     | 22.66     |
| **TrOCR** | **55.38**      | **12.70** | **12.22** |


**TrOCR** vượt trội rõ rệt: CER chỉ bằng ~1/3 so với CRNN/SVTR, nhờ tận dụng pretrained backbone (ViT encoder + Japanese BERT decoder) và khả năng đọc không gian 2D qua Cross-Attention mà không bị ràng buộc bởi hướng quét cố định.

---

## Ứng dụng Demo

Demo được xây dựng theo kiến trúc **client-server**:

```
┌─────────────────────┐         Gradio Public URL          ┌──────────────────────┐
│   React Frontend    │ ◄──────────────────────────────────►│   Kaggle GPU Backend │
│   (Docker + Nginx)  │         @gradio/client              │   (demo-cs231.ipynb) │
│   localhost:8080    │                                     │  SVTR / CRNN / TrOCR │
└─────────────────────┘                                     └──────────────────────┘
```

### Frontend

- **Stack**: React + TypeScript + Vite + TailwindCSS
- **UI**: Lucide React, Framer Motion
- Đóng gói bằng **Docker + Nginx**

### Backend

- Notebook Python chạy trên Kaggle GPU, khởi tạo **Gradio server** với `share=True` tạo endpoint công khai
- Xử lý toàn bộ pipeline: Text Detection (DBNet++) → Snippet Isolation → Text Recognition

### Hai chế độ vận hành (Pipeline Mode)


| Chế độ               | Đầu vào                          | Mô tả                                                                      |
| -------------------- | -------------------------------- | -------------------------------------------------------------------------- |
| **Recognition Only** | Ảnh đã cắt sẵn (snippet)         | Chỉ nhận dạng văn bản, so sánh trực tiếp CRNN / SVTR / TrOCR trên cùng ảnh |
| **Full Pipeline**    | Trang manga nguyên gốc (JPG/PNG) | Detection → cắt snippet → nhận dạng, sát với điều kiện thực tế             |


### Triển khai nhanh

**Bước 1** — Chạy backend trên Kaggle:

```bash
# Mở Demo/demo-cs231.ipynb trên Kaggle, chạy toàn bộ cell
# Cell cuối in ra:
# Running on public URL: https://xxxxxxxxxxxxxxxx.gradio.live
```

**Bước 2** — Build và chạy frontend bằng Docker:

```bash
cd Demo
docker build --build-arg VITE_GRADIO_URL="https://xxxxxxxxxxxxxxxx.gradio.live" -t manga-ocr-frontend .
docker run -d -p 8080:80 manga-ocr-frontend
# Truy cập: http://localhost:8080
```

> Chi tiết đầy đủ xem tại `[Demo/README.md](Demo/README.md)`.

---

## Cấu trúc Repository

```
CS231/
├── notebooks/
│   ├── final_data.ipynb           # Xây dựng bubble_dataset và line_dataset từ Manga109-s
│   ├── final_train_crnn.ipynb     # Huấn luyện CRNN trên RunPod (PaddleOCR)
│   ├── final_train_svtr.ipynb     # Huấn luyện SVTR trên RunPod (PaddleOCR)
│   ├── trocr-rec.ipynb            # Fine-tune TrOCR trên Kaggle (PyTorch + LoRA)
│   └── final_eda_line_dataset.ipynb  # EDA bộ dữ liệu line
│
├── Demo/
│   ├── demo-cs231.ipynb           # Backend Gradio chạy trên Kaggle GPU
│   ├── src/App.tsx                # Giao diện React chính
│   ├── Dockerfile                 # Production build (Nginx)
│   ├── Dockerfile.dev             # Development build
│   └── README.md                  # Hướng dẫn triển khai Demo
│
├── app/
│   ├── infer.py                   # Entry point inference pipeline
│   ├── preprocess.py              # Tiền xử lý ảnh (bubble → line)
│   ├── paddle_rec.py              # Nhận dạng với CRNN/SVTR (PaddleOCR)
│   ├── postprocess.py             # Hậu xử lý kết quả
│   └── utils.py                   # Tiện ích chung
│
├── models/
│   ├── crnn_mobile_line/          # Artifact CRNN (weights + inference model)
│   ├── svtr_tiny_line/            # Artifact SVTR-Tiny (weights + inference model)
│   └── trocr/                     # Link tải TrOCR checkpoint
│
├── configs/
│   ├── infer.crnn.yaml            # Config inference CRNN
│   ├── infer.svtr.yaml            # Config inference SVTR
│   └── infer.default.yaml         # Config mặc định
│
├── controls/
│   ├── run_infer.sh               # Script chạy inference (Linux/macOS)
│   └── run_infer.ps1              # Script chạy inference (Windows PowerShell)
│
├── input/bubble/                  # Ảnh test mẫu
├── archives/                      # Notebook thực nghiệm, log, output mẫu
├── docs/                          # Tài liệu bổ sung
├── CS231.pdf                      # Báo cáo đề tài
└── requirements.txt               # Python dependencies
```

---

## Tài liệu tham khảo chính

- **CRNN**: Shi et al., *An End-to-End Trainable Neural Network for Image-based Sequence Recognition*, 2015
- **SVTR**: Du et al., *SVTR: Scene Text Recognition with a Single Visual Model*, IJCAI 2022
- **TrOCR**: Li et al., *TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models*, 2022
- **LoRA**: Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022
- **Manga109-s**: [manga109.org](http://www.manga109.org/en/)
- **Japanese BERT**: `cl-tohoku/bert-base-japanese-char-v2`, Tohoku NLP Group

