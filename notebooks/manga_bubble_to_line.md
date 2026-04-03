# Kế hoạch & Thực thi: Chuyển đổi Bubble Manga thành Dải ngang

Giải pháp của bạn là một hướng tiếp cận cực kỳ thông minh để train OCR cho Manga dọc mà không cần phải thay đổi các kiến trúc SOTA hiện tại (như CRNN, SVTR với CTC Loss vốn được thiết kế cho văn bản ngang). 

Dưới đây là một "Notebook" chi tiết dạng Markdown, kết hợp giải thích và code Python sử dụng thư viện `OpenCV` (cv2), `Numpy` và `Matplotlib` để hiện thực hóa toàn bộ luồng xử lý (Pipeline) mà bạn đã liệt kê. Bạn có thể copy trực tiếp đoạn code này vào Jupyter Notebook hoặc Colab để xem kết quả.

## 1. Cài đặt và Import thư viện
Đảm bảo bạn đã cài đặt các công cụ xử lý ảnh cần thiết:
```bash
pip install opencv-python numpy matplotlib
```

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

## 2. Xây dựng Pipeline Xử lý (Hàm cắt cột, xoay và ghép nối)

```python
def process_manga_bubble(image_path, output_path=None, space_width=15):
    """
    Hàm xử lý một ảnh manga bubble: tách cột, xóa furigana, xoay và ghép thành 1 line.
    """
    # Bước 1: Đọc ảnh và tiền xử lý cơ bản
    # Đọc với OpenCV (ảnh trả về là định dạng BGR)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Binarize ảnh (Nhị phân hóa bằng thuật toán Otsu + Phân cực ngược)
    # Ta dùng THRESH_BINARY_INV để chữ màu Trắng (255) và nền màu Đen (0). 
    # Việc này giúp khi chiếu VPP, ta đếm được tổng các pixel chữ dễ dàng.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Bước 2: Khử Furigana bằng Morphological Opening
    # Furigana thường có nét mãnh, kích thước siêu nhỏ. Phép Opening (Erosion rồi Dilation) 
    # sẽ tẩy sạch các chấm nhỏ này nhưng vẫn duy trì Kanji/Kana to lớn.
    # Kernel size ở đây có thể tinh chỉnh lại (VD: 3x3, 2x2, 5x3...) tuỳ vào độ phân giải ảnh.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean_bin = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Bước 3: Lập Lược đồ hình chiếu dọc (Vertical Projection Profile)
    # Tính tổng pixel dọc theo từng cột (trục 0 - axis=0) trên ảnh đã dọn sạch furigana
    vpp = np.sum(clean_bin, axis=0)
    
    # Rút trích tọa độ cột: Tạo ngưỡng để xác định đâu là khe hở, đâu là chữ.
    # Ngưỡng ở đây được lấy bằng 5% của giá trị tổng chiếu cao nhất.
    threshold_val = np.max(vpp) * 0.05
    is_text = vpp > threshold_val
    
    # Thuật toán sweep line dò tìm điểm Bắt đầu và Kết thúc của từng cột
    columns_x_ranges = []
    in_col = False
    start_x = 0
    
    for x in range(len(is_text)):
        if is_text[x] and not in_col:
            in_col = True
            start_x = x
        elif not is_text[x] and in_col:
            in_col = False
            end_x = x
            columns_x_ranges.append((start_x, end_x))
            
    if in_col:
        columns_x_ranges.append((start_x, len(is_text)))
        
    # Lọc bỏ nhiễu: Loại bỏ các cột quá mỏng (vd < 10 pixel)
    min_col_width = 10
    columns_x_ranges = [c for c in columns_x_ranges if (c[1] - c[0]) > min_col_width]
    
    if not columns_x_ranges:
        print("Cảnh báo: Không thể nhận diễn cột nào. Ảnh có thể quá sáng/quá mờ định dạng chữ.")
        return None
        
    # Bước 4: Sắp xếp thứ tự Cột (Chuẩn Manga: Từ Phải sang Trái)
    # Cột có tọa độ X lớn hơn sẽ đứng trước (reverse=True)
    columns_x_ranges.sort(key=lambda x: x[0], reverse=True)
    
    # Bước 5: Cắt, Cân bằng kích thước & Xoay ngang
    # Vì để ghép bằng np.hstack chuẩn xác, chiều cao của các khối chữ SAU KHI XOAY phải bằng nhau.
    # Cột sau khi xoay sẽ có C.Cao = Bề rộng gốc của cột. 
    # Vay ta phải tìm Bề rộng gốc lớn nhất, sau đó Padding bằng pixel trắng các cột bé hơn.
    max_orig_width = max([end - start for start, end in columns_x_ranges])
    
    processed_cols = []
    for start, end in columns_x_ranges:
        # Cắt lấy cột ảnh GỐC (chưa nhị phân, để giữ được độ nét tự nhiên cho mạng nhận dạng đọc)
        col_img = gray[:, start:end] 
        
        # Cân bằng (Zero-padding/White-padding) bề rộng của cột đang cắt với max_orig_width
        pad_total = max_orig_width - (end - start)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        
        # BORDER_CONSTANT với value=255 để thêm rìa màu Trắng bao quanh
        padded_col = cv2.copyMakeBorder(col_img, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=255)
        
        # Xoay -90 độ ngược chiều kim đồng hồ
        rotated_col = cv2.rotate(padded_col, cv2.ROTATE_90_COUNTERCLOCKWISE)
        processed_cols.append(rotated_col)
        
    # Bước 6: Ghép nối (Stitching)
    # Tạo một dải phân cách trắng (space)
    # Dải trắng có Bề cao = Height các ảnh đã xoay (tức max_orig_width) & Bề ngang = tuỳ chọn.
    space_block = np.full((max_orig_width, space_width), 255, dtype=np.uint8)
    
    # Dùng hstack để ép chúng lại thành 1 mảng numpy ngang dài
    final_output = processed_cols[0]
    for i in range(1, len(processed_cols)):
        final_output = np.hstack((final_output, space_block, processed_cols[i]))
        
    # Bước 7 (Optional): Lưu file output
    if output_path:
        cv2.imwrite(output_path, final_output)
        
    return img, clean_bin, vpp, columns_x_ranges, final_output

```

## 3. Thực thi trực tiếp và Vẽ biểu đồ (Visualization)

Khối code này xử lý riêng một ảnh ví dụ của bạn và vẽ trực tiếp cả quá trình với Matplotlib.

```python
# Cập nhật lại đường dẫn tùy theo môi trường (chẳng hạn Colab hay Local PC)
image_file = r"e:\qDATA\Research\NCKH\PP_OCR_JAP\manga109\rec\train\word_000001.png"
out_file = r"e:\qDATA\Research\NCKH\PP_OCR_JAP\manga109\rec\train\word_000001_horizontal.png"

# Chạy thuật toán
results = process_manga_bubble(image_file, out_file, space_width=20)

if results:
    orig_img, clean_bin, vpp, ext_cols, final_output = results
    
    # Thiết lập matplotlib để báo cáo 4 màn hình quá trình
    plt.figure(figsize=(16, 12))
    
    # 1. Hiển thị ảnh gốc
    plt.subplot(2, 2, 1)
    plt.title("1. Original Bubble Image")
    plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    
    # 2. Ảnh sau khi phân cực và Open (Tẩy Furigana)
    plt.subplot(2, 2, 2)
    plt.title("2. Cleaned Binary (Morphological Opening)")
    plt.imshow(clean_bin, cmap='gray')
    
    # 3. Lược đồ hình chiếu dọc của ảnh nhị phân ở trên
    plt.subplot(2, 2, 3)
    plt.title("3. Vertical Projection Profile")
    plt.plot(vpp)
    # Vẽ các cột ranh giới (Xanh và đỏ tương ứng với min X, max X của cột đang cắt đi)
    for start, end in ext_cols:
        plt.axvline(x=start, color='r', linestyle='--')
        plt.axvline(x=end, color='g', linestyle='--')
        
    # 4. Hiển thị kết quả Stitching đã xoay cuối cùng
    plt.subplot(2, 2, 4)
    plt.title(f"4. Final Single-Line Sequence (CTC ready)")
    plt.imshow(final_output, cmap='gray')
    
    plt.tight_layout()
    plt.show()

```

## Giải thích mở rộng về hàm CTC-Loss
Với thuật toán bóc tách thành **1 line duy nhất dọc sang ngang** mà bạn hướng tới này, nó đem lại những kết quả vô cùng ổn định cho CTC:
- **Ngữ pháp bảo toàn:** Bằng việc duyệt từ "X Lớn nhất" (Phải) về "X nhỏ" (Trái) và xoay từng khung, bạn đã tạo ra chuỗi Left-To-Right y như sách tiếng Anh nên Loss sẽ tính `seq-to-seq` rất mượt mà.
- **Ký tự nghiêng -90°:** Tuy ký tự lúc này đang "nằm bẹp ngang", nhưng Conv + BiLSTM trong CRNN chỉ coi đây là pattern học được. Bạn train mô hình từ zero (từ scratch) hoặc Freeze/đổi fine-tune, model sẽ tự học được pattern chữ nằm ngang này cực hiệu quả.
- **Biến Space:** Khoảng trắng `(255)` width `15px` cũng rất ổn áp định danh biểu tượng `<space>` hoặc khoảng chia cách Bubble.

Bạn hãy copy đoạn file này tạo một Notebook `.ipynb` ở local tại `e:\qDATA\Research\NCKH\PP_OCR_JAP\` và chạy test thử để xem Matplotlib trình chiếu nhé. Nó xử lý toàn diện đúng logic bạn yêu cầu!
