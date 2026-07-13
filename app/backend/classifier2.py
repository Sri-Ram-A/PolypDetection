from pathlib import Path

import numpy as np
import cv2

import tensorrt as trt
import pycuda.driver as cuda

from PIL import Image

from backend import cuda_ctx

BASE_DIR = Path(__file__).resolve().parents[1]
ENGINE_PATH = BASE_DIR / "models" / "best_squeezenet_fp16.engine"

INPUT_NAME = "input"
OUTPUT_NAME = "output"
IN_H, IN_W = 224, 224

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_NAMES = ["esophagitis", "polyps", "ulcerative-colitis"]


class SqueezeNetClassifier:
    def __init__(self, engine_path: Path = ENGINE_PATH, class_names=CLASS_NAMES):

        self.class_names = class_names

        cuda_ctx.push()
        try:
            self.logger = trt.Logger(trt.Logger.WARNING)

            with open(engine_path, "rb") as f:
                runtime = trt.Runtime(self.logger)
                self.engine = runtime.deserialize_cuda_engine(f.read())

            self.context = self.engine.create_execution_context()

            self._inp = np.zeros((1, 3, IN_H, IN_W), dtype=np.float32)
            self._out = np.zeros((1, len(self.class_names)), dtype=np.float32)

            self.d_in = cuda.mem_alloc(self._inp.nbytes)
            self.d_out = cuda.mem_alloc(self._out.nbytes)
            self.stream = cuda.Stream()
        finally:
            cuda_ctx.pop()

    def _softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def _preprocess(self, image):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        image = image.convert("RGB").resize((IN_W, IN_H))
        arr = np.array(image).astype(np.float32) / 255.0
        normed = (arr - IMAGENET_MEAN) / IMAGENET_STD
        chw = normed.transpose(2, 0, 1)[None].astype(np.float32)

        return np.ascontiguousarray(chw)

    def _infer(self, img_np):
        cuda_ctx.push()
        try:
            cuda.memcpy_htod_async(self.d_in, img_np, self.stream)
            self.context.set_tensor_address(INPUT_NAME, int(self.d_in))
            self.context.set_tensor_address(OUTPUT_NAME, int(self.d_out))
            self.context.execute_async_v3(self.stream.handle)
            cuda.memcpy_dtoh_async(self._out, self.d_out, self.stream)
            self.stream.synchronize()
            return self._out[0].copy()
        finally:
            cuda_ctx.pop()

    def predict(self, image):
        x = self._preprocess(image)
        logits = self._infer(x)
        probs = self._softmax(logits)
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])

        return {"class_name": self.class_names[idx], "confidence": confidence}

    def is_polyp(self, image, threshold=0.8):
        POLYPS_CLASSES = ["polyps"]
        result = self.predict(image)

        return result["class_name"] in POLYPS_CLASSES and result["confidence"] >= threshold