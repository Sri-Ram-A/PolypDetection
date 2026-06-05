from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from backend.classifier import DenseNetClassifier
from backend.segmenter import PraNetSegmenter

class VideoProcessor:

    def __init__(self):

        self.classifier = DenseNetClassifier()
        self.segmenter = PraNetSegmenter()
        
    def process_video(self, input_video_path, output_video_path):
        
        cap = cv2.VideoCapture(
            str(input_video_path)
        )

        if not cap.isOpened():

            raise ValueError(
                f"Cannot open {input_video_path}"
            )
            
        fps = int(
            cap.get(
                cv2.CAP_PROP_FPS
            )
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )
        
        output_width = width * 2
        output_height = height
        
        fourcc = cv2.VideoWriter_fourcc(
            *"avc1"
        )

        writer = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            fps,
            (
                output_width,
                output_height
            )
        )
        
        polyp_frames = 0
        processed_frames = 0
        class_counts = {}
        
        confidence_history = []
        
        pbar = tqdm(
            total=total_frames,
            desc="Processing Video"
        )

        while True:

            ret, frame = cap.read()

            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )
            
            result = self.classifier.predict(
                rgb_frame
            )

            class_name = result[
                "class_name"
            ]

            confidence = result[
                "confidence"
            ]
            
            confidence_history.append(confidence)
            
            class_counts[class_name] = (
                class_counts.get(class_name, 0) + 1
            )
            
            is_polyp = self.classifier.is_polyp(
                rgb_frame
            )
            
            if is_polyp:

                polyp_frames += 1

                mask, overlay = (
                    self.segmenter.predict_and_overlay(
                        rgb_frame
                    )
                )

                overlay = cv2.resize(
                    overlay,
                    (
                        width,
                        height
                    )
                )

            else:

                overlay = rgb_frame.copy()
                
            status = (
                "POLYP DETECTED"
                if is_polyp
                else "NO POLYP"
            )
            
            timestamp = processed_frames / fps
            
            cv2.putText(
                overlay,
                f"Time: {timestamp:.1f}s",
                (20,200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255,255,255),
                2
            )            

            status_color = (
                (0,255,0)
                if is_polyp
                else (0,0,255)
            )
                
            
            cv2.putText(
                overlay,
                f"Frame: {processed_frames}",
                (20,120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255,255,255),
                2
            )
            
            cv2.putText(
                overlay,
                status,
                (20,160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                status_color,
                2
            )

            cv2.putText(
                overlay,
                f"{confidence:.2%}",
                (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )
            
            overlay = cv2.cvtColor(
                overlay,
                cv2.COLOR_RGB2BGR
            )
            
            combined = np.hstack(
                [
                    frame,
                    overlay
                ]
            )
            
            writer.write(
                combined
            )

            processed_frames += 1

            pbar.update(1)
            
        pbar.close()

        cap.release()

        writer.release()
        
        class_counts = dict(
            sorted(
                class_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
        )
        
        dominant_class = max(
            class_counts,
            key=class_counts.get
        )
        
        return {
            "frames_processed":
                processed_frames,

            "polyp_frames":
                polyp_frames,

            "polyp_percentage":
                (
                    polyp_frames
                    /
                    max(processed_frames, 1)
                )
                * 100,
                
            "class_counts": class_counts,
            "dominant_class": dominant_class,

            "average_confidence": float(np.mean(
                confidence_history
            )),
            "median_confidence": float(np.median(
                confidence_history
            )),
            "min_confidence": float(np.min(
                confidence_history
            )),
            "max_confidence": float(np.max(
                confidence_history
            )),
            "output_video_path": str(output_video_path)
        }