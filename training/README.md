# SiteGuard AI — Model Training & Optimization Guide

Optimized for **NVIDIA GeForce MX550 (2 GB VRAM)** and standard GPUs.

---

## ⚡ Fast Training on 2GB VRAM (MX550)

To achieve maximum F1-score, confidence, and speed without encountering CUDA Out-of-Memory (OOM):

```bash
# Run 2GB VRAM-optimized training
python train.py --model rtdetr-l.pt --epochs 50 --batch 2 --accumulate 4 --imgsz 640
```

### Why these settings work best for MX550 2GB:
1. **Physical Batch Size = 2 + Gradient Accumulation = 4**: Simulates an effective batch size of 8 while fitting completely inside 2 GB VRAM.
2. **Automatic Mixed Precision (AMP FP16)**: Cuts GPU memory footprint in half and uses Turing Tensor cores for 2x faster matrix multiplications.
3. **Cosine Learning Rate Decay (`cos_lr=True`)**: Leads to faster epoch convergence, avoiding loss plateaus and maximizing the final F1 score.
4. **Data Augmentations**: Mosaic (1.0) and HSV color jitter give rich occlusion invariance for small PPE items (hard hats, masks) without requiring extra dataset collection.

---

## 📊 Evaluation & Validation

To evaluate on validation or test split and generate precision, recall, F1, and mAP:

```bash
python evaluate.py --weights ../backend/models/best.pt --split val
```
