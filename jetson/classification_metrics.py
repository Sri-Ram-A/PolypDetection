# classify_eval.py — SqueezeNet Classification Full Evaluation
import os, glob, time
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from sklearn.metrics import (confusion_matrix, classification_report,
                              accuracy_score, precision_recall_fscore_support,
                              roc_auc_score, log_loss)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax

ENGINE_PATH   = "best_squeezenet_fp16.engine"
DATA_DIR      = "/workspace/BDT/data/raw/Kvasir-classify/kvasir-dataset"
SELECTED_CLASSES = ["esophagitis", "polyps", "ulcerative-colitis"]
H, W          = 224, 224

# ── Load engine ───────────────────────────────────────────────────────────────
logger  = trt.Logger(trt.Logger.WARNING)
with open(ENGINE_PATH, "rb") as f:
    engine  = trt.Runtime(logger).deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

inp_np  = np.zeros((1, 3, H, W), dtype=np.float32)
out_np  = np.zeros((1, len(SELECTED_CLASSES)), dtype=np.float32)
d_in    = cuda.mem_alloc(inp_np.nbytes)
d_out   = cuda.mem_alloc(out_np.nbytes)
stream  = cuda.Stream()

# ── Preprocess ────────────────────────────────────────────────────────────────
def preprocess(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W, H)).astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return img.transpose(2, 0, 1)[None].astype(np.float32)

# ── Infer ─────────────────────────────────────────────────────────────────────
def infer(img_np):
    cuda.memcpy_htod_async(d_in, np.ascontiguousarray(img_np), stream)
    context.set_tensor_address("input",  int(d_in))
    context.set_tensor_address("output", int(d_out))
    context.execute_async_v3(stream.handle)
    cuda.memcpy_dtoh_async(out_np, d_out, stream)
    stream.synchronize()
    return out_np[0].copy()

# ── Collect images ────────────────────────────────────────────────────────────
image_paths, gt_labels = [], []
for idx, cls in enumerate(SELECTED_CLASSES):
    cls_dir = os.path.join(DATA_DIR, cls)
    paths   = sorted(glob.glob(f"{cls_dir}/*.jpg") + glob.glob(f"{cls_dir}/*.png"))
    image_paths.extend(paths)
    gt_labels.extend([idx] * len(paths))
    print(f"  {cls:<25} → {len(paths)} images")

gt_labels = np.array(gt_labels)

# ── Warmup ────────────────────────────────────────────────────────────────────
dummy = preprocess(image_paths[0])
for _ in range(3):
    infer(dummy)

# ── Inference loop ────────────────────────────────────────────────────────────
logits_all, pred_labels, latencies = [], [], []

for path in image_paths:
    inp     = preprocess(path)
    t0      = time.perf_counter()
    logits  = infer(inp)
    latencies.append(time.perf_counter() - t0)
    logits_all.append(logits)
    pred_labels.append(int(np.argmax(logits)))

logits_all  = np.array(logits_all)
probs_all   = softmax(logits_all, axis=1)          # for ROC-AUC & log loss
pred_labels = np.array(pred_labels)

# ── Metrics ───────────────────────────────────────────────────────────────────
acc        = accuracy_score(gt_labels, pred_labels)
prec, rec, f1, _ = precision_recall_fscore_support(
                        gt_labels, pred_labels, average='macro', zero_division=0)
prec_per, rec_per, f1_per, sup = precision_recall_fscore_support(
                        gt_labels, pred_labels, average=None, zero_division=0)
roc_auc    = roc_auc_score(gt_labels, probs_all, multi_class='ovr', average='macro')
logloss    = log_loss(gt_labels, probs_all)
cm         = confusion_matrix(gt_labels, pred_labels)
mean_lat   = np.mean(latencies)
fps        = 1.0 / mean_lat

# ── Print report ──────────────────────────────────────────────────────────────
print("\n" + "="*62)
print("  SqueezeNet Classification — Full Evaluation Report")
print("="*62)
print(f"  Engine        : {ENGINE_PATH}")
print(f"  Images tested : {len(image_paths)}")
print(f"  Classes       : {SELECTED_CLASSES}")
print("-"*62)
print(f"  Accuracy      : {acc:.4f}")
print(f"  Precision     : {prec:.4f}  (macro)")
print(f"  Recall        : {rec:.4f}  (macro)")
print(f"  F1 Score      : {f1:.4f}  (macro)")
print(f"  ROC-AUC       : {roc_auc:.4f}  (OvR macro)")
print(f"  Log Loss      : {logloss:.4f}")
print("-"*62)
print(f"  {'Class':<25} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
print(f"  {'-'*53}")
for i, cls in enumerate(SELECTED_CLASSES):
    print(f"  {cls:<25} {prec_per[i]:>6.4f} {rec_per[i]:>6.4f} {f1_per[i]:>6.4f} {sup[i]:>8}")
print("-"*62)
print(f"  Latency (mean): {mean_lat*1000:.2f} ms/image")
print(f"  FPS           : {fps:.1f}")
print("="*62)

# ── Confusion matrix plot ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('#1a1a1a')

sns.heatmap(cm, annot=True, fmt='d', cmap='plasma',
            xticklabels=SELECTED_CLASSES, yticklabels=SELECTED_CLASSES,
            linewidths=2, linecolor='#1a1a1a', ax=axes[0],
            cbar_kws={'label': 'Count'})
axes[0].set_xlabel('Predicted', fontsize=12, fontweight='bold', color='white')
axes[0].set_ylabel('Actual',    fontsize=12, fontweight='bold', color='white')
axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold',
                  pad=12, color='white')
axes[0].tick_params(colors='white')
axes[0].set_facecolor('#1a1a1a')

# normalised CM
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='plasma',
            xticklabels=SELECTED_CLASSES, yticklabels=SELECTED_CLASSES,
            linewidths=2, linecolor='#1a1a1a', ax=axes[1],
            cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1)
axes[1].set_xlabel('Predicted', fontsize=12, fontweight='bold', color='white')
axes[1].set_ylabel('Actual',    fontsize=12, fontweight='bold', color='white')
axes[1].set_title('Normalised Confusion Matrix', fontsize=14, fontweight='bold',
                  pad=12, color='white')
axes[1].tick_params(colors='white')
axes[1].set_facecolor('#1a1a1a')

plt.tight_layout()
plt.savefig('classify_eval_cm.png', dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
plt.show()
print("Confusion matrix saved → classify_eval_cm.png")