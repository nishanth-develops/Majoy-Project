"""
Apple Detection and Tracking System using YOLOv8 and DeepSORT
-----------------------------------------------------------

System Requirements:
- CPU: Minimum 4 cores, recommended 8 cores
- RAM: Minimum 8GB, recommended 16GB
- GPU: Minimum 4GB VRAM, recommended 8GB VRAM (NVIDIA GPU with CUDA support)
- Storage: Minimum 10GB free space
- OS: Windows 10/11, Ubuntu 20.04+, or macOS 12+

Performance Notes:
- GPU processing achieves ~20-30 FPS on recommended hardware
- CPU-only processing achieves ~5-10 FPS on recommended hardware
- Processing speed varies based on video resolution and object count

Setup Instructions:
1. Create virtual environment: python -m venv venv
2. Activate virtual environment:
   - Windows: venv\\Scripts\\activate
   - Unix/macOS: source venv/bin/activate
3. Install requirements: pip install -r requirements.txt
4. Download YOLOv8 weights: yolo download yolov8n.pt
5. Run: python apple_counter.py --video path/to/video.mp4
"""

import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
import time
from ultralytics import YOLO
from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config
import logging
import sys

class AppleTracker:
    def __init__(self, video_path, output_path=None, confidence=0.3, show_display=True):
        """
        Initialize the Apple Tracking System
        
        Args:
            video_path (str): Path to input video file
            output_path (str, optional): Path to save output video
            confidence (float): Detection confidence threshold
            show_display (bool): Whether to show real-time display
        """
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # System checks
        self.check_system_requirements()
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")
            
        # Video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Setup output video writer if specified
        self.writer = None
        if output_path:
            self.writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                self.fps,
                (self.frame_width, self.frame_height)
            )
        
        # Initialize models
        self.setup_models(confidence)
        
        # Tracking variables
        self.total_apples = 0
        self.tracked_apples = {}
        self.show_display = show_display
        
        self.logger.info("Apple Tracker initialized successfully")

    def setup_logging(self):
        """Configure logging settings"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('apple_tracker.log')
            ]
        )

    def check_system_requirements(self):
        """Check if system meets minimum requirements"""
        self.logger.info("Checking system requirements...")
        
        # Check CUDA availability
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cpu':
            self.logger.warning("CUDA not available. Running on CPU will be slower")
        
        # Check available memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.total < 8 * (1024 ** 3):  # 8GB
                self.logger.warning("Less than 8GB RAM detected. Performance may be impacted")
        except ImportError:
            self.logger.warning("Could not check system memory")

    def setup_models(self, confidence):
        """Initialize YOLO and DeepSORT models"""
        # Initialize YOLO
        try:
            self.yolo = YOLO('yolov8n.pt')  # Using nano model for better performance
            self.logger.info("YOLO model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading YOLO model: {e}")
            raise

        # Initialize DeepSORT
        try:
            cfg = get_config()
            cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
            self.deepsort = DeepSort(
                cfg.DEEPSORT.REID_CKPT,
                max_dist=cfg.DEEPSORT.MAX_DIST,
                min_confidence=confidence,
                nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                max_age=cfg.DEEPSORT.MAX_AGE,
                n_init=cfg.DEEPSORT.N_INIT,
                nn_budget=cfg.DEEPSORT.NN_BUDGET,
                use_cuda=(self.device == 'cuda')
            )
            self.logger.info("DeepSORT initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing DeepSORT: {e}")
            raise

    def process_frame(self, frame):
        """
        Process a single frame for apple detection and tracking
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            tuple: Processed frame and apple count
        """
        # Run YOLO detection
        results = self.yolo(frame)[0]
        
        # Process detections
        detections = []
        confidences = []
        bboxes = []
        
        # Filter for apple detections (class 47 in COCO dataset)
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = r
            if int(cls) == 47:  # Apple class
                detections.append([x1, y1, x2, y2, conf])
                confidences.append(conf)
                bboxes.append([x1, y1, x2 - x1, y2 - y1])
        
        if len(detections) > 0:
            # Update DeepSORT tracker
            tracks = self.deepsort.update(
                np.array(bboxes),
                np.array(confidences),
                frame
            )
            
            # Update tracking information
            current_tracks = set()
            for track in tracks:
                track_id = int(track[4])
                current_tracks.add(track_id)
                
                if track_id not in self.tracked_apples:
                    self.tracked_apples[track_id] = {
                        'first_seen': time.time(),
                        'counted': False
                    }
                
                # Draw tracking visualization
                self.draw_tracking(frame, track)
                
                # Count new apples
                if not self.tracked_apples[track_id]['counted']:
                    if time.time() - self.tracked_apples[track_id]['first_seen'] > 1.0:  # Confidence delay
                        self.total_apples += 1
                        self.tracked_apples[track_id]['counted'] = True
            
            # Clean up old tracks
            self.clean_up_tracks(current_tracks)
        
        # Add count to frame
        self.draw_stats(frame)
        
        return frame

    def draw_tracking(self, frame, track):
        """Draw tracking visualization on frame"""
        bbox = track[:4]
        track_id = int(track[4])
        
        # Draw bounding box
        cv2.rectangle(
            frame,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 255, 0),
            2
        )
        
        # Draw ID and confidence
        cv2.putText(
            frame,
            f"Apple #{track_id}",
            (int(bbox[0]), int(bbox[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    def draw_stats(self, frame):
        """Draw statistics on frame"""
        # Add total count
        cv2.putText(
            frame,
            f"Total Apples: {self.total_apples}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Add FPS
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
        """Remove old tracks that are no longer visible"""
        tracked_ids = list(self.tracked_apples.keys())
        for track_id in tracked_ids:
            if track_id not in current_tracks:
                if time.time() - self.tracked_apples[track_id]['first_seen'] > 5.0:  # Remove after 5 seconds
                    del self.tracked_apples[track_id]

    def process_video(self):
        """Process the entire video"""
        frame_count = 0
        processing_times = []
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            start_time = time.time()
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Update FPS
            if frame_count % 30 == 0:
                self.fps = 1.0 / np.mean(processing_times[-30:])
            
            # Write frame
            if self.writer:
                self.writer.write(processed_frame)
            
            # Display frame
            if self.show_display:
                cv2.imshow('Apple Tracking', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            
            # Log progress
            if frame_count % 100 == 0:
                self.logger.info(f"Processed {frame_count} frames. Current apple count: {self.total_apples}")
        
        # Cleanup
        self.cleanup()
        
        return self.total_apples

    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
        self.logger.info(f"Processing complete. Final apple count: {self.total_apples}")

def main():
    parser = argparse.ArgumentParser(description='Apple Detection and Tracking System')
    parser.add_argument('--video', required=True, help='Path to input video file')
    parser.add_argument('--output', help='Path to output video file')
    parser.add_argument('--confidence', type=float, default=0.3, help='Detection confidence threshold')
    parser.add_argument('--no-display', action='store_true', help='Disable display window')
    
    args = parser.parse_args()
    
    tracker = AppleTracker(
        args.video,
        args.output,
        args.confidence,
        not args.no_display
    )
    
    total_apples = tracker.process_video()
    print(f"\nFinal Apple Count: {total_apples}")

if __name__ == "__main__":
    main()

#python apple_counter.py --video path/to/video.mp4 --output output.mp4 --confidence 0.3 use this for execution.
