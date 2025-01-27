# !pip install ultralytics torch opencv-python numpy deep-sort-realtime psutil
# !mkdir -p /content/apple_tracking


import cv2
import numpy as np
import torch
from pathlib import Path
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import logging
import sys
import psutil
from google.colab.patches import cv2_imshow
from google.colab import files
from IPython.display import display, HTML

class AppleTracker:
    def __init__(self, video_path, output_path=None, confidence=0.3):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.check_system_requirements()
        
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
            
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Changed from CAP_PROP_HEIGHT
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        self.writer = None
        if output_path:
            self.writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                self.fps,
                (self.frame_width, self.frame_height)
            )
        
        self.setup_models(confidence)
        self.total_apples = 0
        self.tracked_apples = {}

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )

    def check_system_requirements(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            self.logger.warning("Running on CPU - processing will be slower")
        
        memory = psutil.virtual_memory()
        if memory.total < 8 * (1024 ** 3):
            self.logger.warning("Less than 8GB RAM detected")

    def setup_models(self, confidence):
        try:
            self.yolo = YOLO('yolov8n.pt')
        except Exception as e:
            self.logger.error(f"YOLO model loading failed: {e}")
            raise

        try:
            self.deepsort = DeepSort(
                max_age=5,
                n_init=2,
                nms_max_overlap=1.0,
                max_cosine_distance=0.3,
                nn_budget=None,
                override_track_class=None,
                embedder="mobilenet",
                half=True,
                bgr=True,
                embedder_gpu=True if self.device == 'cuda' else False
            )
        except Exception as e:
            self.logger.error(f"DeepSORT initialization failed: {e}")
            raise

    def process_frame(self, frame):
        try:
            results = self.yolo(frame)[0]
        except Exception as e:
            self.logger.error(f"YOLO inference failed: {e}")
            return frame
        
        detections = []
        
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = r
            if int(cls) == 47:  # Apple class
                detections.append(([x1, y1, x2, y2], conf, 'apple'))
        
        if detections:
            tracks = self.deepsort.update_tracks(detections, frame=frame)
            current_tracks = set()
            
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                current_tracks.add(track_id)
                ltrb = track.to_ltrb()
                
                if track_id not in self.tracked_apples:
                    self.tracked_apples[track_id] = {
                        'first_seen': time.time(),
                        'counted': False
                    }
                
                self.draw_tracking(frame, ltrb, track_id)
                
                if not self.tracked_apples[track_id]['counted']:
                    if time.time() - self.tracked_apples[track_id]['first_seen'] > 1.0:
                        self.total_apples += 1
                        self.tracked_apples[track_id]['counted'] = True
            
            self.clean_up_tracks(current_tracks)
        
        self.draw_stats(frame)
        return frame

    def draw_tracking(self, frame, bbox, track_id):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Apple #{track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    def draw_stats(self, frame):
        cv2.putText(
            frame,
            f"Total Apples: {self.total_apples}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    def clean_up_tracks(self, current_tracks):
        tracked_ids = list(self.tracked_apples.keys())
        for track_id in tracked_ids:
            if track_id not in current_tracks:
                if time.time() - self.tracked_apples[track_id]['first_seen'] > 5.0:
                    del self.tracked_apples[track_id]

    def process_video(self):
        frame_count = 0
        processing_times = []
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            start_time = time.time()
            processed_frame = self.process_frame(frame)
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            if frame_count % 30 == 0:
                self.fps = 1.0 / np.mean(processing_times[-30:])
            
            if self.writer:
                self.writer.write(processed_frame)
            
            # Display frame in Colab
            if frame_count % 30 == 0:  # Show every 30th frame
                cv2_imshow(processed_frame)
                from IPython.display import clear_output
                clear_output(wait=True)
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames. Current count: {self.total_apples}")
        
        self.cleanup()
        return self.total_apples

    def cleanup(self):
        self.cap.release()
        if self.writer:
            self.writer.release()
        print(f"Final apple count: {self.total_apples}")


#block 3333
from google.colab import files
import os

print("Please upload your video file:")
uploaded = files.upload()

video_path = next(iter(uploaded))
output_path = '/content/apple_tracking/output.mp4'

tracker = AppleTracker(video_path, output_path, confidence=0.3)
total_apples = tracker.process_video()

print(f"\nProcessing complete!")
print(f"Final Apple Count: {total_apples}")