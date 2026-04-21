#!/usr/bin/env bash
# controls/setup.sh
# -----------------------------------------------------------------------------
# Bootstrap môi trường Linux (Runpod / Ubuntu) để chạy notebook:
#   notebooks/final_train_svtr.ipynb   (sinh từ notebooks/_gen_final_train_svtr_nb.py)
#
# Script làm 4 việc, idempotent (chạy lại nhiều lần không hỏng):
#   1. Cài tool hệ thống  : git, wget, tar, curl, ca-certificates, bzip2, build-essential
#   2. Cài Miniconda       : vào ~/miniconda3 nếu chưa có
#   3. Tạo env `paddleocr` : Python 3.10, đăng ký Jupyter kernel `jpocr`
#   4. Cài pip deps tối thiểu mà notebook cần TRƯỚC khi tự nó cài tiếp
#      (numpy, pandas, pyyaml, tqdm, gdown, ipykernel)
#
# Những thứ KHÔNG cài ở đây (notebook tự cài bên trong các cell):
#   - paddlepaddle-gpu==3.0.0            (Bước 1)
#   - PaddleOCR + requirements.txt       (Bước 2)
#   - gdown (cài lại cho chắc)           (Bước 4)
#
# Cách dùng (1 lần trên máy fresh):
#   bash controls/setup.sh
#
# Sau khi script xong:
#   - Restart Jupyter
#   - Chọn kernel  "jpocr"
#   - Mở & chạy notebook notebooks/final_train_svtr.ipynb
# -----------------------------------------------------------------------------

set -euo pipefail

log()  { printf "\n[setup] %s\n" "$*"; }
warn() { printf "\n[setup][WARN] %s\n" "$*" >&2; }

# -----------------------------------------------------------------------------
# 0. Chuẩn bị sudo prefix (Runpod chạy root -> không cần sudo)
# -----------------------------------------------------------------------------
if [[ "${EUID}" -eq 0 ]]; then
    SUDO=""
else
    SUDO="sudo"
fi

# -----------------------------------------------------------------------------
# 1. Tool hệ thống
# -----------------------------------------------------------------------------
log "Step 1/4: install system packages (git, wget, tar, curl, ...)"
export DEBIAN_FRONTEND=noninteractive
${SUDO} apt-get update -y
${SUDO} apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    tar \
    bzip2 \
    ca-certificates \
    build-essential

# -----------------------------------------------------------------------------
# 2. Miniconda
# -----------------------------------------------------------------------------
MINICONDA_DIR="${HOME}/miniconda3"
MINICONDA_SH="/tmp/Miniconda3-latest-Linux-x86_64.sh"

if [[ -x "${MINICONDA_DIR}/bin/conda" ]]; then
    log "Step 2/4: Miniconda already installed at ${MINICONDA_DIR}, skip."
else
    log "Step 2/4: install Miniconda to ${MINICONDA_DIR}"
    wget -q --show-progress -O "${MINICONDA_SH}" \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    # -b = batch (silent, accept license), -p = prefix
    bash "${MINICONDA_SH}" -b -p "${MINICONDA_DIR}"
    rm -f "${MINICONDA_SH}"
fi

# Nạp conda vào shell hiện tại (script) để dùng `conda activate`
# shellcheck disable=SC1091
source "${MINICONDA_DIR}/etc/profile.d/conda.sh"

# Đảm bảo shell đăng nhập sau cũng có conda (idempotent append)
if ! grep -q "miniconda3/etc/profile.d/conda.sh" "${HOME}/.bashrc" 2>/dev/null; then
    log "Append 'source conda.sh' into ~/.bashrc"
    echo "source ${MINICONDA_DIR}/etc/profile.d/conda.sh" >> "${HOME}/.bashrc"
fi

# -----------------------------------------------------------------------------
# 3. Env paddleocr + Jupyter kernel
# -----------------------------------------------------------------------------
ENV_NAME="paddleocr"
KERNEL_NAME="paddleocr"
KERNEL_DISPLAY="jpocr"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    log "Step 3/4: env '${ENV_NAME}' already exists, skip creation."
else
    log "Step 3/4: create conda env '${ENV_NAME}' (python=3.10)"
    # Không dùng -y: một số cài đặt conda không cho phép; có thể hỏi xác nhận trên terminal.
    conda create -n "${ENV_NAME}" python=3.10
fi

log "Activate env '${ENV_NAME}'"
conda activate "${ENV_NAME}"

log "Upgrade pip/setuptools/wheel"
python -m pip install -q --upgrade pip setuptools wheel

log "Install ipykernel + register Jupyter kernel '${KERNEL_DISPLAY}'"
python -m pip install -q ipykernel
python -m ipykernel install --user \
    --name "${KERNEL_NAME}" \
    --display-name "${KERNEL_DISPLAY}"

# -----------------------------------------------------------------------------
# 4. Pip deps tối thiểu để chạy được các cell đầu của notebook
#    (notebook sẽ tự cài thêm paddlepaddle-gpu + PaddleOCR/requirements.txt)
# -----------------------------------------------------------------------------
log "Step 4/4: install base pip deps"
python -m pip install -q \
    "numpy>=1.21" \
    "pandas>=1.3" \
    "PyYAML>=5.1" \
    "tqdm>=4.64" \
    "gdown>=4.5"

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
log "DONE."
cat <<EOF

Tiếp theo:
  1. RESTART Jupyter (để kernel '${KERNEL_DISPLAY}' hiện ra trong kernel picker).
  2. Mở notebook notebooks/final_train_svtr.ipynb.
  3. Chọn kernel '${KERNEL_DISPLAY}' (tương ứng conda env '${ENV_NAME}').
  4. Run All. Các bước Bước 1 (paddlepaddle-gpu) và Bước 2 (clone PaddleOCR +
     cài PaddleOCR/requirements.txt) sẽ tự chạy trong notebook.

Shell session mới:
  Dùng 'conda activate ${ENV_NAME}' là đủ (đã append vào ~/.bashrc).
EOF
