# test_seg.py — HarDNet-MSEG Full Evaluation
import os, glob, time
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from scipy.ndimage import distance_transform_edt

ENGINE_PATH = "hardnet_mseg_fp16.engine"
IMAGE_DIR   = "/workspace/BDT/data/raw/kvasir-seg/Kvasir-SEG/images"
MASK_DIR    = "/workspace/BDT/data/raw/kvasir-seg/Kvasir-SEG/masks"
OUT_DIR     = "seg_results"
H, W        = 352, 352
IOU_THRESH  = 0.5          # threshold for mAP TP/FP decision
BOUNDARY_DW = 2            # boundary width in pixels for Boundary F1
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load engine ───────────────────────────────────────────────────────────────
logger  = trt.Logger(trt.Logger.WARNING)
with open(ENGINE_PATH, "rb") as f:
    engine  = trt.Runtime(logger).deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

inp    = np.zeros((1, 3, H, W), dtype=np.float32)
out    = np.zeros((1, 1, H, W), dtype=np.float32)
d_in   = cuda.mem_alloc(inp.nbytes)
d_out  = cuda.mem_alloc(out.nbytes)
stream = cuda.Stream()

# ── Preprocess ────────────────────────────────────────────────────────────────
def preprocess(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W, H)).astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return img.transpose(2, 0, 1)[None].astype(np.float32)

# ── Infer ─────────────────────────────────────────────────────────────────────
def infer(img_np):
    cuda.memcpy_htod_async(d_in, np.ascontiguousarray(img_np), stream)
    context.set_tensor_address("image", int(d_in))
    context.set_tensor_address("mask",  int(d_out))
    context.execute_async_v3(stream.handle)
    cuda.memcpy_dtoh_async(out, d_out, stream)
    stream.synchronize()
    return 1 / (1 + np.exp(-out[0, 0]))

# ── Metric helpers ────────────────────────────────────────────────────────────
def get_boundary(mask_bin, dw=BOUNDARY_DW):
    """Extract boundary pixels via distance transform."""
    dist  = distance_transform_edt(mask_bin)
    dist2 = distance_transform_edt(1 - mask_bin)
    return ((dist <= dw) & (mask_bin == 1)) | ((dist2 <= dw) & (mask_bin == 0) & (dist2 > 0))

def boundary_f1(pred_bin, gt_bin):
    pb = get_boundary(pred_bin).astype(np.uint8)
    gb = get_boundary(gt_bin).astype(np.uint8)
    tp = (pb & gb).sum()
    if tp == 0:
        return 0.0
    p  = tp / (pb.sum() + 1e-6)
    r  = tp / (gb.sum() + 1e-6)
    return 2 * p * r / (p + r + 1e-6)

def boundary_iou(pred_bin, gt_bin):
    pb = get_boundary(pred_bin).astype(np.uint8)
    gb = get_boundary(gt_bin).astype(np.uint8)
    inter = (pb & gb).sum()
    union = (pb | gb).sum()
    return inter / (union + 1e-6)

def iou_score(pred_bin, gt_bin):
    inter = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum()
    return inter / (union + 1e-6)

def dice_score(pred_bin, gt_bin):
    inter = (pred_bin & gt_bin).sum()
    return 2 * inter / (pred_bin.sum() + gt_bin.sum() + 1e-6)

def segmentation_quality(pred_bin, gt_bin):
    """SQ component of Panoptic Quality = IoU of matched pair."""
    return iou_score(pred_bin, gt_bin)

def recognition_quality(pred_bin, gt_bin, iou_thresh=IOU_THRESH):
    """RQ component = F1 based on TP/FP/FN at given IoU threshold."""
    iou = iou_score(pred_bin, gt_bin)
    tp  = 1 if iou >= iou_thresh else 0
    fp  = 1 - tp
    fn  = 1 - tp
    return tp / (tp + 0.5 * fp + 0.5 * fn + 1e-6)

def panoptic_quality(pred_bin, gt_bin):
    sq = segmentation_quality(pred_bin, gt_bin)
    rq = recognition_quality(pred_bin, gt_bin)
    return sq * rq, sq, rq

def average_precision_single(pred_prob, gt_bin,
                              thresholds=np.arange(0.5, 1.0, 0.05)):
    """
    mAP over IoU thresholds [0.50 : 0.05 : 0.95] (COCO-style).
    For binary segmentation each image has exactly one instance,
    so AP = mean of {1 if IoU>=t else 0} over thresholds.
    """
    aps = []
    for t in thresholds:
        pred_bin = (pred_prob > 0.5).astype(np.uint8)
        aps.append(1.0 if iou_score(pred_bin, gt_bin) >= t else 0.0)
    return float(np.mean(aps))

# ── Collect images ────────────────────────────────────────────────────────────
image_paths = sorted(glob.glob(f"{IMAGE_DIR}/*.jpg") +
                     glob.glob(f"{IMAGE_DIR}/*.png"))
mask_paths  = sorted(glob.glob(f"{MASK_DIR}/*.jpg")  +
                     glob.glob(f"{MASK_DIR}/*.png"))

assert len(image_paths) == len(mask_paths) and len(image_paths) > 0, \
    "Image / mask count mismatch or empty directory."

# ── Per-image accumulators ────────────────────────────────────────────────────
ious, dices, bious, bf1s, pqs, sqs, rqs, aps = [], [], [], [], [], [], [], []
latencies = []

# ── Warmup (3 passes so CUDA is hot before timing) ───────────────────────────
dummy = preprocess(image_paths[0])
for _ in range(3):
    infer(dummy)

# ── Main eval loop ────────────────────────────────────────────────────────────
for img_path, mask_path in zip(image_paths, mask_paths):
    name = os.path.basename(img_path)

    # load GT
    gt_raw  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    gt_bin  = (cv2.resize(gt_raw, (W, H)) > 127).astype(np.uint8)

    # infer + time
    inp_np  = preprocess(img_path)
    t0      = time.perf_counter()
    prob    = infer(inp_np)
    latencies.append(time.perf_counter() - t0)

    pred_bin = (prob > 0.5).astype(np.uint8)

    # metrics
    iou  = iou_score(pred_bin,  gt_bin)
    dice = dice_score(pred_bin, gt_bin)
    biou = boundary_iou(pred_bin, gt_bin)
    bf1  = boundary_f1(pred_bin,  gt_bin)
    pq, sq, rq = panoptic_quality(pred_bin, gt_bin)
    ap   = average_precision_single(prob, gt_bin)

    ious.append(iou);  dices.append(dice)
    bious.append(biou); bf1s.append(bf1)
    pqs.append(pq);   sqs.append(sq);  rqs.append(rq)
    aps.append(ap)

    # save mask
    cv2.imwrite(f"{OUT_DIR}/{name}", (pred_bin * 255).astype(np.uint8))

# ── Aggregate ─────────────────────────────────────────────────────────────────
mean_lat = np.mean(latencies)
fps      = 1.0 / mean_lat

print("\n" + "="*60)
print("  HarDNet-MSEG — Full Evaluation Results")
print("="*60)
print(f"  Images evaluated       : {len(image_paths)}")
print(f"  Resolution             : {H}×{W}")
print("-"*60)
print(f"  IoU  (Jaccard)         : {np.mean(ious):.4f}  ± {np.std(ious):.4f}")
print(f"  Dice (F1)              : {np.mean(dices):.4f}  ± {np.std(dices):.4f}")
print("-"*60)
print(f"  Boundary IoU           : {np.mean(bious):.4f}  ± {np.std(bious):.4f}")
print(f"  Boundary F1            : {np.mean(bf1s):.4f}  ± {np.std(bf1s):.4f}")
print("-"*60)
print(f"  Panoptic Quality  (PQ) : {np.mean(pqs):.4f}  ± {np.std(pqs):.4f}")
print(f"  Segmentation Q.   (SQ) : {np.mean(sqs):.4f}  ± {np.std(sqs):.4f}")
print(f"  Recognition Q.    (RQ) : {np.mean(rqs):.4f}  ± {np.std(rqs):.4f}")
print("-"*60)
print(f"  mAP  (IoU 0.5:0.95)    : {np.mean(aps):.4f}  ± {np.std(aps):.4f}")
print("-"*60)
print(f"  Latency (mean)         : {mean_lat*1000:.2f} ms/image")
print(f"  FPS                    : {fps:.1f}")
print("="*60)
print(f"\nMasks saved to → {OUT_DIR}/")