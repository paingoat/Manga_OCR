# %% [markdown]
# # Fine-tune TrOCR với Japanese BERT Decoder + LoRA trên Manga109-s (H100 Optimized)
# 
# **Kiến trúc mới (đã sửa)**:
# - **Encoder**: ViT từ `microsoft/trocr-base-printed` (giữ nguyên pretrained weights)
# - **Decoder**: `cl-tohoku/bert-base-japanese-char-v2` (pretrained Japanese BERT — không phải random weights)
# - **LoRA**: Chỉ train ~2-3% tham số trên attention layers
# 
# **Tại sao thay đổi**:
# Cách cũ (swap tokenizer vào TrOCR decoder) → decoder bắt đầu từ **random embedding** với English-pretrained weights → kết quả hỗn loạn.
# Cách mới dùng JP BERT làm decoder thực sự → decoder có **prior knowledge về tiếng Nhật** ngay từ đầu.

# %% [markdown]
# ## 1. Cài đặt dependencies

# %%
!pip install -q transformers datasets evaluate jiwer accelerate peft bitsandbytes
!pip install -q fugashi unidic-lite  # required for cl-tohoku tokenizer


# %% [markdown]
# ## 2. Import thư viện + tối ưu H100

# %%
import os
import random
import torch
import numpy as np
from PIL import Image
from datasets import Dataset
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    default_data_collator,
    set_seed,
    )
import evaluate
from tqdm import tqdm
from peft import LoraConfig, get_peft_model

# ==================== REPRODUCIBILITY ====================
SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==================== H100 OPTIMIZATIONS ====================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print('H100 optimizations enabled (TF32 + bf16)')
print(f'Seed: {SEED}')

print('GPU available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))


# %% [markdown]
# ## 3. Tải Manga109-s

# %%
from huggingface_hub import snapshot_download

HF_TOKEN = ''
BASE_DIR = '/kaggle/working/manga109s_dataset'
os.makedirs(BASE_DIR, exist_ok=True)

print('📥 Đang tải Manga109-s...')
snapshot_download(
    repo_id='hal-utokyo/Manga109-s',
    repo_type='dataset',
    local_dir=BASE_DIR,
    token=HF_TOKEN,
)
print(f'✅ Tải xong. Files: {os.listdir(BASE_DIR)}')


# %% [markdown]
# ## 4. Giải nén dataset

# %%
import zipfile

ZIP_PATH     = '/kaggle/working/manga109s_dataset/Manga109s_released_2023_12_07.zip'
EXTRACT_PATH = '/kaggle/working/manga109_extracted'
DATASET_ROOT = f'{EXTRACT_PATH}/Manga109s_released_2023_12_07'

if not os.path.exists(DATASET_ROOT) or not os.listdir(DATASET_ROOT):
    print('📦 Giải nén Manga109-s...')
    os.makedirs(EXTRACT_PATH, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(EXTRACT_PATH)
    print('✅ Giải nén xong!')
else:
    print('✅ Đã giải nén trước đó, bỏ qua.')

print(f'Nội dung: {os.listdir(DATASET_ROOT)}')


# %% [markdown]
# ## 5. Tạo REC dataset (crop text box)

# %%
import random
import xml.etree.ElementTree as ET
from typing import Optional

class Manga109RecDatasetCreator:
    def __init__(self, dataset_root: str):
        self.dataset_root    = dataset_root
        self.annotations_dir = os.path.join(dataset_root, 'annotations')
        self.images_dir      = os.path.join(dataset_root, 'images')

    def create_rec_dataset(
        self,
        output_dir: str = '/kaggle/working/train_data',
        min_width: int = 20,
        min_height: int = 20,
        max_samples: Optional[int] = 50000,
        train_ratio: float = 0.9,
        seed: int = 42,
    ):
        random.seed(seed)
        rec_dir = os.path.join(output_dir, 'rec')
        train_img_dir = os.path.join(rec_dir, 'train')
        val_img_dir   = os.path.join(rec_dir, 'val')
        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(val_img_dir,   exist_ok=True)

        train_gt = os.path.join(rec_dir, 'rec_gt_train.txt')
        val_gt   = os.path.join(rec_dir, 'rec_gt_val.txt')

        xml_files = sorted(f for f in os.listdir(self.annotations_dir) if f.endswith('.xml'))
        print(f'🔍 Tìm thấy {len(xml_files)} truyện manga.')

        all_samples = []
        for xml_file in tqdm(xml_files, desc='Parsing manga'):
            book_title = os.path.splitext(xml_file)[0]
            xml_path   = os.path.join(self.annotations_dir, xml_file)
            try:
                root = ET.parse(xml_path).getroot()
            except:
                continue

            for page in root.findall('.//page'):
                page_idx = page.get('index')
                img_filename = f'{int(page_idx):03d}.jpg'
                img_path = os.path.join(self.images_dir, book_title, img_filename)
                if not os.path.exists(img_path):
                    continue
                try:
                    img = Image.open(img_path).convert('RGB')
                except:
                    continue

                for text_obj in page.findall('text'):
                    try:
                        xmin, ymin = int(text_obj.get('xmin')), int(text_obj.get('ymin'))
                        xmax, ymax = int(text_obj.get('xmax')), int(text_obj.get('ymax'))
                        label = (text_obj.text or '').strip().replace('\n', ' ').replace('\t', ' ')
                        if not label:
                            continue
                        if (xmax - xmin) < min_width or (ymax - ymin) < min_height:
                            continue
                        all_samples.append((img.crop((xmin, ymin, xmax, ymax)), label))
                    except:
                        continue

                if max_samples and len(all_samples) >= max_samples * 2:
                    break

        random.shuffle(all_samples)
        if max_samples:
            all_samples = all_samples[:max_samples]

        split_idx = int(len(all_samples) * train_ratio)
        train_samples = all_samples[:split_idx]
        val_samples   = all_samples[split_idx:]
        print(f'📊 Split: {len(train_samples):,} train | {len(val_samples):,} val')

        def save_split(samples, img_dir, gt_path, split_name, start_idx=1):
            with open(gt_path, 'w', encoding='utf-8') as f:
                for i, (crop, label) in enumerate(tqdm(samples, desc=f'Saving {split_name}'), start_idx):
                    img_name = f'word_{i:06d}.png'
                    crop.save(os.path.join(img_dir, img_name))
                    f.write(f'{split_name}/{img_name}\t{label}\n')

        save_split(train_samples, train_img_dir, train_gt, 'train')
        save_split(val_samples,   val_img_dir,   val_gt,   'val', len(train_samples)+1)
        print('✅ Dataset REC hoàn tất!')

creator = Manga109RecDatasetCreator(DATASET_ROOT)
creator.create_rec_dataset(
    output_dir   = '/kaggle/working/train_data',
    max_samples  = 50000,
    train_ratio  = 0.9,
    seed         = 42
)


# %%
import os
import re
import unicodedata
from datasets import Dataset

REC_ROOT = '/kaggle/working/train_data/rec'

TRAIN_IMG_DIR = os.path.join(REC_ROOT, 'train')
VAL_IMG_DIR   = os.path.join(REC_ROOT, 'val')
TRAIN_GT      = os.path.join(REC_ROOT, 'rec_gt_train.txt')
VAL_GT        = os.path.join(REC_ROOT, 'rec_gt_val.txt')

MIN_CHARS = 2
MAX_CHARS = 80

def normalize_jp_text(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('\u3000', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_gt_file(gt_path, img_base_dir):
    samples = []
    dropped_empty = 0
    dropped_short = 0
    dropped_long = 0
    dropped_missing_img = 0
    dropped_parse_error = 0

    with open(gt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rel_path, text = line.split('\t', 1)
                img_path = os.path.join(img_base_dir, os.path.basename(rel_path))
                if not os.path.exists(img_path):
                    dropped_missing_img += 1
                    continue

                text = normalize_jp_text(text)
                if not text:
                    dropped_empty += 1
                    continue
                if len(text) < MIN_CHARS:
                    dropped_short += 1
                    continue
                if len(text) > MAX_CHARS:
                    dropped_long += 1
                    continue

                samples.append({'img_path': img_path, 'text': text})
            except Exception:
                dropped_parse_error += 1

    stats = {
        'kept': len(samples),
        'dropped_empty': dropped_empty,
        'dropped_short': dropped_short,
        'dropped_long': dropped_long,
        'dropped_missing_img': dropped_missing_img,
        'dropped_parse_error': dropped_parse_error,
    }
    return samples, stats

train_samples, train_stats = load_gt_file(TRAIN_GT, TRAIN_IMG_DIR)
val_samples, val_stats     = load_gt_file(VAL_GT,   VAL_IMG_DIR)

train_dataset = Dataset.from_list(train_samples)
val_dataset   = Dataset.from_list(val_samples)

print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,}")
print('Train stats:', train_stats)
print('Val stats:  ', val_stats)


# %% [markdown]
# ## 6. Load Model: TrOCR Encoder + Japanese BERT Decoder
# 
# > **Đây là thay đổi cốt lõi so với notebook cũ.**
# 
# Thay vì swap tokenizer vào decoder của TrOCR (khiến decoder bắt đầu từ random embedding),
# ta dùng `from_encoder_decoder_pretrained()` để ghép:
# - **Encoder**: ViT của TrOCR — đã pretrain nhận dạng ảnh văn bản
# - **Decoder**: `cl-tohoku/bert-base-japanese-char-v2` — pretrained Japanese BERT thực sự
# 
# Cross-attention layers (kết nối encoder-decoder) sẽ được khởi tạo ngẫu nhiên và cần được train.
# LoRA sẽ cover cả hai phần này.
# 

# %%
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    BertJapaneseTokenizer,
    VisionEncoderDecoderModel,
    ViTModel,                        # load encoder trực tiếp
    BertLMHeadModel,                 # load decoder trực tiếp
    VisionEncoderDecoderConfig,
)
from peft import get_peft_model, LoraConfig

ENCODER_MODEL = 'microsoft/trocr-base-printed'
DECODER_MODEL = 'cl-tohoku/bert-base-japanese-char-v2'

# ================== IMAGE PROCESSOR + TOKENIZER ==================
print('🖼️  Loading image processor...')
feature_extractor = AutoImageProcessor.from_pretrained(ENCODER_MODEL)

print('🔤 Loading Japanese tokenizer...')
jp_tokenizer = BertJapaneseTokenizer.from_pretrained(DECODER_MODEL)
print(f'   Vocab size: {len(jp_tokenizer)}')

# Verify tokenizer
sample = 'その手紙は届いた'
print(f"   Sample: '{sample}' → {jp_tokenizer.tokenize(sample)}")

# ================== LOAD ENCODER & DECODER RIÊNG ==================
# Lấy encoder ViT từ bên trong TrOCR
print('🏗️  Loading TrOCR encoder (ViT)...')
trocr_full = VisionEncoderDecoderModel.from_pretrained(ENCODER_MODEL)
encoder = trocr_full.encoder   # ViTModel đã pretrain
del trocr_full                 # giải phóng memory

# Load JP BERT làm decoder — bật is_decoder + cross_attention
print('🏗️  Loading JP BERT decoder...')
decoder = BertLMHeadModel.from_pretrained(
    DECODER_MODEL,
    is_decoder=True,
    add_cross_attention=True,
)

# ================== GHÉP LẠI ==================
print('🔗 Assembling VisionEncoderDecoderModel...')
model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)


# ================== CẤU HÌNH DECODER ==================
model.config.decoder_start_token_id = jp_tokenizer.cls_token_id
model.config.pad_token_id           = jp_tokenizer.pad_token_id
model.config.eos_token_id           = jp_tokenizer.sep_token_id
model.config.decoder.vocab_size     = len(jp_tokenizer)
model.config.use_cache              = False

# FIX: set thêm vào generation_config — model.config và generation_config là 2 object riêng
model.generation_config.decoder_start_token_id = jp_tokenizer.cls_token_id
model.generation_config.pad_token_id           = jp_tokenizer.pad_token_id
model.generation_config.eos_token_id           = jp_tokenizer.sep_token_id
model.generation_config.bos_token_id           = jp_tokenizer.cls_token_id  # fallback
model.generation_config.max_new_tokens         = 64
model.generation_config.early_stopping         = True
model.generation_config.no_repeat_ngram_size   = 0
model.generation_config.num_beams              = 4

print(f'✅ Model ready')
print(f'   Encoder params: {sum(p.numel() for p in model.encoder.parameters()):,}')
print(f'   Decoder params: {sum(p.numel() for p in model.decoder.parameters()):,}')

# %% [markdown]
# ## 7. Áp dụng LoRA (ổn định hơn)
# 
# `target_modules` dùng `query/value` để khớp cả ViT encoder và BERT decoder attention.
# Tăng `lora_dropout` để regularize tốt hơn, giảm nguy cơ train loss xuống nhanh nhưng val loss/CER không cải thiện.
# 

# %%
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias='none',
    # BERT decoder dùng query/value (không dùng q_proj/v_proj như LLaMA-style modules)
    target_modules=[
        'query', 'value',
    ],
    task_type='SEQ_2_SEQ_LM',
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print('LoRA applied')


# %% [markdown]
# ## 8. Tạo dataset cho Trainer

# %%
from PIL import Image

MAX_TARGET_LENGTH = 96  # tăng nhẹ để giảm cắt cụt text box dài

def preprocess(example):
    # Encode ảnh qua ViT feature extractor
    image = Image.open(example['img_path']).convert('RGB')
    pixel_values = feature_extractor(image, return_tensors='pt').pixel_values.squeeze(0)

    # Chuẩn hóa text trước khi tokenize để train/eval nhất quán
    clean_text = normalize_jp_text(example['text'])
    tokenized = jp_tokenizer(
        clean_text,
        padding='max_length',
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        return_attention_mask=False,
    )
    labels = tokenized.input_ids

    # Mask padding để không tính vào loss
    labels = [l if l != jp_tokenizer.pad_token_id else -100 for l in labels]
    return {'pixel_values': pixel_values, 'labels': labels}

train_dataset = train_dataset.shuffle(seed=42)
val_dataset   = val_dataset.shuffle(seed=42)

print('Preprocessing train set...')
train_dataset = train_dataset.map(preprocess, remove_columns=['img_path', 'text'])
print('Preprocessing val set...')
val_dataset   = val_dataset.map(preprocess,   remove_columns=['img_path', 'text'])

train_dataset.set_format(type='torch', columns=['pixel_values', 'labels'])
val_dataset.set_format(type='torch',   columns=['pixel_values', 'labels'])

print('Dataset ready')
print(f'   Train: {len(train_dataset):,} | Val: {len(val_dataset):,}')
print(f'   MAX_TARGET_LENGTH: {MAX_TARGET_LENGTH}')


# %% [markdown]
# ## 9. Training Arguments (H100 Optimized)

# %%
import inspect

# Tương thích nhiều phiên bản Transformers, đồng thời đảm bảo eval/save match khi load_best_model_at_end=True
sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
strategy_kwargs = {}
gc_kwargs = {}

# Trên một số bản torch/transformers + PEFT, gradient checkpointing (reentrant) gây lỗi
# "A different number of tensors was saved during forward and recomputation".
# Mặc định tắt để train ổn định hơn trên Kaggle. Nếu cần bật lại, đổi thành True.
ENABLE_GRADIENT_CHECKPOINTING = False

# Một số version dùng eval_strategy, một số dùng evaluation_strategy.
# Nếu có cả hai thì set cùng giá trị để tránh lệch.
if 'eval_strategy' in sig.parameters:
    strategy_kwargs['eval_strategy'] = 'epoch'
if 'evaluation_strategy' in sig.parameters:
    strategy_kwargs['evaluation_strategy'] = 'epoch'

if not strategy_kwargs:
    raise RuntimeError('Không tìm thấy tham số eval strategy phù hợp trong Seq2SeqTrainingArguments')

# Khi bật gradient checkpointing, ưu tiên non-reentrant để tránh mismatch tensor
if ENABLE_GRADIENT_CHECKPOINTING and 'gradient_checkpointing_kwargs' in sig.parameters:
    gc_kwargs['gradient_checkpointing_kwargs'] = {'use_reentrant': False}

training_args = Seq2SeqTrainingArguments(
    output_dir='./trocr-manga109-jpbert-lora',

    # ================== BATCH SIZE & MEMORY ==================
    per_device_train_batch_size=12,
    per_device_eval_batch_size=24,
    gradient_accumulation_steps=4,     # Effective batch size = 48
    gradient_checkpointing=ENABLE_GRADIENT_CHECKPOINTING,

    # ================== TRAINING PARAMS (STABLE) ==================
    learning_rate=3e-5,
    num_train_epochs=12,
    bf16=True,
    tf32=True,
    max_grad_norm=1.0,
    # IMPORTANT: PEFT + Seq2SeqTrainer + label smoothing có thể làm thiếu decoder_input_ids
    # và gây lỗi: "You must specify exactly one of input_ids or inputs_embeds"
    label_smoothing_factor=0.0,

    # ================== DataLoader ==================
    dataloader_num_workers=4,
    dataloader_prefetch_factor=2,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,

    # ================== Optimizer ==================
    # LoRA params ít, không cần 8-bit optimizer để tránh nhiễu thêm
    optim='adamw_torch_fused',
    weight_decay=0.01,
    lr_scheduler_type='cosine',
    warmup_ratio=0.08,

    # ================== Logging & Eval ==================
    do_eval=True,
    logging_strategy='steps',
    logging_steps=25,
    logging_first_step=True,
    save_strategy='epoch',
    save_total_limit=3,
    load_best_model_at_end=True,

    # ================== Generation ==================
    predict_with_generate=True,
    generation_num_beams=2,
    generation_max_length=96,
    metric_for_best_model='cer',
    greater_is_better=False,

    report_to='none',
    remove_unused_columns=False,
    label_names=['labels'],
    **strategy_kwargs,
    **gc_kwargs,
 )

# Sanity check: load_best_model_at_end yêu cầu eval/save strategy phải khớp.
resolved_eval_strategy = getattr(training_args, 'eval_strategy', None)
if resolved_eval_strategy is None:
    resolved_eval_strategy = getattr(training_args, 'evaluation_strategy', None)

print('Training args ready')
print(f'   Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}')
print(f'   LR: {training_args.learning_rate} | Optim: {training_args.optim}')
print(f'   Label smoothing: {training_args.label_smoothing_factor}')
print(f'   Gradient checkpointing: {training_args.gradient_checkpointing}')
print(f'   Gradient checkpointing kwargs: {gc_kwargs}')
print(f'   Eval strategy resolved: {resolved_eval_strategy}')
print(f'   Save strategy: {training_args.save_strategy}')


# %%
import os
import torch

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()

# Không bật gradient checkpointing thủ công ở đây.
# Nếu training_args.gradient_checkpointing=True, Trainer sẽ tự xử lý enable.
model.config.use_cache = False

if training_args.gradient_checkpointing:
    print('Gradient checkpointing sẽ được bật bởi Trainer (an toàn hơn).')
else:
    print('Gradient checkpointing đang tắt để tránh lỗi checkpoint tensor mismatch.')

model = model.to('cuda')

print('✅ Memory optimization applied')
print(f'Current VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GiB')


# %% [markdown]
# ## 10. Tạo Trainer + Compute Metrics

# %%
import re
import unicodedata
import numpy as np
import evaluate
from transformers import EarlyStoppingCallback

metric_cer = evaluate.load('cer')

def normalize_for_cer(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', '', text)
    return text

def compute_metrics(pred):
    predictions = pred.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Fallback an toàn nếu predictions đang là logits
    if isinstance(predictions, np.ndarray) and predictions.ndim == 3:
        predictions = np.argmax(predictions, axis=-1)

    labels = pred.label_ids.copy()

    # Restore pad tokens để decode
    labels[labels == -100] = jp_tokenizer.pad_token_id

    pred_str  = jp_tokenizer.batch_decode(predictions, skip_special_tokens=True)
    label_str = jp_tokenizer.batch_decode(labels,      skip_special_tokens=True)

    pred_str  = [normalize_for_cer(p) for p in pred_str]
    label_str = [normalize_for_cer(l) for l in label_str]

    cer = metric_cer.compute(predictions=pred_str, references=label_str)
    return {'cer': float(cer)}

# Guard: tránh lỗi decoder_input_ids khi dùng PEFT + label smoothing trong Seq2SeqTrainer
if getattr(training_args, 'label_smoothing_factor', 0.0) and training_args.label_smoothing_factor > 0:
    print('Override label_smoothing_factor -> 0.0 để tránh lỗi decoder input với PEFT')
    training_args.label_smoothing_factor = 0.0

# ================== TRAINER ==================
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=1e-3)],
)

print('Trainer sẵn sàng (CER + early stopping)')


# %% [markdown]
# ## 11. Bắt đầu Train

# %%
import os
import json

train_result = trainer.train()

best_ckpt = trainer.state.best_model_checkpoint
best_metric = trainer.state.best_metric
print('Best checkpoint:', best_ckpt)
print('Best metric (CER):', best_metric)

# ==================== SAVE ARTIFACTS ====================
OUTPUT_DIR = './trocr-manga109-jpbert-lora-final'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Với PEFT/LoRA: save adapter từ model đang được trainer giữ (đã load best model nếu bật load_best_model_at_end)
trainer.model.save_pretrained(OUTPUT_DIR)
jp_tokenizer.save_pretrained(OUTPUT_DIR)
feature_extractor.save_pretrained(OUTPUT_DIR)

# Lưu metadata để inference khôi phục đúng base model + config
save_meta = {
    'encoder_model': ENCODER_MODEL,
    'decoder_model': DECODER_MODEL,
    'max_target_length': MAX_TARGET_LENGTH,
    'best_model_checkpoint': best_ckpt,
    'best_metric_cer': best_metric,
}
with open(os.path.join(OUTPUT_DIR, 'train_meta.json'), 'w', encoding='utf-8') as f:
    json.dump(save_meta, f, ensure_ascii=False, indent=2)

# Optional: merge LoRA vào base model để inference nhanh/đơn giản hơn
MERGED_DIR = f'{OUTPUT_DIR}-merged'
try:
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(MERGED_DIR, safe_serialization=True)
    print(f'Merged model saved to: {MERGED_DIR}')
except Exception as e:
    print(f'Skip merged save (fallback adapter-only): {e}')

print(f'Training hoàn tất! Artifacts đã lưu vào: {OUTPUT_DIR}')


# %% [markdown]
# ## 12. Inference trên Val Set

# %%
import os
import re
import json
import unicodedata
import torch
import evaluate
import pandas as pd
from tqdm import tqdm
from PIL import Image
from peft import PeftModel
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    VisionEncoderDecoderModel,
    BertLMHeadModel,
)

# ==================== PATHS ====================
OUTPUT_DIR = './trocr-manga109-jpbert-lora-final'
OUTPUT_DIR_ABS = os.path.abspath(OUTPUT_DIR)
MERGED_DIR_ABS = os.path.abspath(f'{OUTPUT_DIR}-merged')

print('Loading model artifacts...')

# Load processor/tokenizer đã save từ train
feature_extractor = AutoImageProcessor.from_pretrained(OUTPUT_DIR_ABS, local_files_only=True)
jp_tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR_ABS, local_files_only=True)

# Load metadata để đảm bảo reconstruct đúng base architecture
meta_path = os.path.join(OUTPUT_DIR_ABS, 'train_meta.json')
if os.path.exists(meta_path):
    with open(meta_path, 'r', encoding='utf-8') as f:
        save_meta = json.load(f)
    ENCODER_MODEL = save_meta.get('encoder_model', 'microsoft/trocr-base-printed')
    DECODER_MODEL = save_meta.get('decoder_model', 'cl-tohoku/bert-base-japanese-char-v2')
else:
    ENCODER_MODEL = 'microsoft/trocr-base-printed'
    DECODER_MODEL = 'cl-tohoku/bert-base-japanese-char-v2'

# ==================== LOAD MODEL ====================
if os.path.exists(MERGED_DIR_ABS):
    print(f'Found merged model: {MERGED_DIR_ABS}')
    infer_model = VisionEncoderDecoderModel.from_pretrained(MERGED_DIR_ABS, local_files_only=True)
else:
    print('Merged model not found, loading base + LoRA adapter...')

    # Build base model giống lúc train
    trocr_full = VisionEncoderDecoderModel.from_pretrained(ENCODER_MODEL)
    encoder = trocr_full.encoder
    del trocr_full

    decoder = BertLMHeadModel.from_pretrained(
        DECODER_MODEL,
        is_decoder=True,
        add_cross_attention=True,
    )
    base_model = VisionEncoderDecoderModel(encoder=encoder, decoder=decoder)

    # Load LoRA adapter theo chuẩn PEFT
    infer_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR_ABS, is_trainable=False)

# ==================== GENERATION CONFIG ====================
infer_model.config.decoder_start_token_id = jp_tokenizer.cls_token_id
infer_model.config.pad_token_id = jp_tokenizer.pad_token_id
infer_model.config.eos_token_id = jp_tokenizer.sep_token_id

infer_model.generation_config.decoder_start_token_id = jp_tokenizer.cls_token_id
infer_model.generation_config.bos_token_id = jp_tokenizer.cls_token_id
infer_model.generation_config.pad_token_id = jp_tokenizer.pad_token_id
infer_model.generation_config.eos_token_id = jp_tokenizer.sep_token_id
infer_model.generation_config.max_new_tokens = 96
infer_model.generation_config.num_beams = 2
infer_model.generation_config.early_stopping = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
infer_model.eval()
infer_model.to(device)
print(f'Model loaded. Device: {device}')

def normalize_for_eval(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', '', text)
    return text

# ==================== INFERENCE FUNCTION ====================
def infer_image(image_path, max_new_tokens=96):
    image = Image.open(image_path).convert('RGB')
    pixel_values = feature_extractor(image, return_tensors='pt').pixel_values.to(device)

    with torch.no_grad():
        generated_ids = infer_model.generate(
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            num_beams=2,
            early_stopping=True,
            no_repeat_ngram_size=0,
        )

    predicted_text = jp_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return normalize_for_eval(predicted_text)

# ==================== EVAL ON VAL SUBSET ====================
VAL_IMG_DIR = '/kaggle/working/train_data/rec/val'
VAL_GT = '/kaggle/working/train_data/rec/rec_gt_val.txt'

val_samples_inf = []
with open(VAL_GT, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rel_path, text = line.split('\t', 1)
        img_path = os.path.join(VAL_IMG_DIR, os.path.basename(rel_path))
        if os.path.exists(img_path):
            val_samples_inf.append({
                'image_path': img_path,
                'ground_truth': normalize_for_eval(text),
            })

print(f'Val samples available: {len(val_samples_inf)}')

metric_cer = evaluate.load('cer')
results = []

for sample in tqdm(val_samples_inf[:50], desc='Inference val set'):
    pred = infer_image(sample['image_path'])
    gt = sample['ground_truth']
    results.append({
        'image': os.path.basename(sample['image_path']),
        'ground_truth': gt,
        'prediction': pred,
        'correct': gt == pred,
    })

df = pd.DataFrame(results)
print('\nInference results (first 50 val images):')
display(df)

acc = df['correct'].mean() * 100
print(f'Exact-match accuracy: {acc:.2f}%')

cer = metric_cer.compute(
    predictions=df['prediction'].tolist(),
    references=df['ground_truth'].tolist(),
)
print(f'CER on 50 samples: {cer:.4f} ({cer*100:.1f}%)')

csv_path = './trocr_inference_val_results.csv'
df.to_csv(csv_path, index=False, encoding='utf-8')
print(f'Results saved to: {csv_path}')



