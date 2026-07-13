from pathlib import Path

import cv2
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda

from PIL import Image

from backend import cuda_ctx

BASE_DIR = Path(__file__).resolve().parents[1]
ENGINE_PATH = BASE_DIR / "models" / "hardnet_mseg_fp16.engine"

INPUT_NAME = "image"
OUTPUT_NAME = "mask"
IN_H, IN_W = 352, 352

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class HardNetMSEGSegmenter:
    def __init__(self, engine_path: Path = ENGINE_PATH):

        cuda_ctx.push()
        try:
            self.logger = trt.Logger(trt.Logger.WARNING)

            with open(engine_path, "rb") as f:
                runtime = trt.Runtime(self.logger)
                self.engine = runtime.deserialize_cuda_engine(f.read())

            self.context = self.engine.create_execution_context()

            self._inp = np.zeros((1, 3, IN_H, IN_W), dtype=np.float32)
            self._out = np.zeros((1, 1, IN_H, IN_W), dtype=np.float32)

            self.d_in = cuda.mem_alloc(self._inp.nbytes)
            self.d_out = cuda.mem_alloc(self._out.nbytes)
            self.stream = cuda.Stream()
        finally:
            cuda_ctx.pop()

    def _preprocess(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)

        resized = cv2.resize(image, (IN_W, IN_H)).astype(np.float32) / 255.0
        normed = (resized - IMAGENET_MEAN) / IMAGENET_STD
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
            return 1 / (1 + np.exp(-self._out[0, 0]))
        finally:
            cuda_ctx.pop()

    def predict_mask(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)

        x = self._preprocess(image)
        pred = self._infer(x)
        binary_mask = (pred > 0.5).astype(np.uint8)

        return binary_mask

    def create_overlay(self, rgb_img, mask, alpha=0.4):
        overlay = rgb_img.copy()
        color_mask = np.zeros_like(rgb_img)
        color_mask[:, :, 1] = mask * 255
        overlay = cv2.addWeighted(rgb_img, 1 - alpha, color_mask, alpha, 0)

        return overlay

    def predict_and_overlay(self, image):
        mask = self.predict_mask(image)
        image = cv2.resize(image, (IN_W, IN_H))
        overlay = self.create_overlay(image, mask)

        return mask, overlay