# test_seg.py
import os, glob, numpy as np, cv2, tensorrt as trt, pycuda.driver as cuda
import pycuda.autoinit

ENGINE_PATH = "hardnet_mseg_fp16.engine"
IMAGE_DIR   = "data/raw/kvasir-seg/Kvasir-SEG/images"
OUT_DIR     = "seg_results"
H, W        = 352, 352
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load engine ──────────────────────────────────────────────────────────────
logger = trt.Logger(trt.Logger.WARNING)
with open(ENGINE_PATH, "rb") as f:
    runtime = trt.Runtime(logger)
    engine  = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

# ── Allocate buffers ─────────────────────────────────────────────────────────
inp  = np.zeros((1, 3, H, W), dtype=np.float32)
out  = np.zeros((1, 1, H, W), dtype=np.float32)
d_in  = cuda.mem_alloc(inp.nbytes)
d_out = cuda.mem_alloc(out.nbytes)
stream = cuda.Stream()

# ── Preprocess ───────────────────────────────────────────────────────────────
def preprocess(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W, H)).astype(np.float32) / 255.0
    img = (img - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
    return img.transpose(2,0,1)[None].astype(np.float32)   # 1x3xHxW

# ── Infer ─────────────────────────────────────────────────────────────────────
# ── Infer (TensorRT 10 API) ───────────────────────────────────────────────────
def infer(img_np):
    cuda.memcpy_htod_async(d_in, np.ascontiguousarray(img_np), stream)
    context.set_tensor_address("image", int(d_in))
    context.set_tensor_address("mask",  int(d_out))
    context.execute_async_v3(stream.handle)
    cuda.memcpy_dtoh_async(out, d_out, stream)
    stream.synchronize()
    return 1 / (1 + np.exp(-out[0, 0]))

# ── Run on first 5 images ─────────────────────────────────────────────────────
images = sorted(glob.glob(f"{IMAGE_DIR}/*.jpg"))[:5]
for path in images:
    name   = os.path.basename(path)
    mask   = infer(preprocess(path))
    binary = (mask > 0.5).astype(np.uint8) * 255
    cv2.imwrite(f"{OUT_DIR}/{name}", binary)
    print(f"{name}  max={mask.max():.3f}  mean={mask.mean():.3f}")

print(f"\nDone — masks saved to {OUT_DIR}/")