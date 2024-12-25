import cv2
import torch
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import json
import threading
import queue
import time

class WasteDetectorDeployment:
    def __init__(self, model_path, confidence_threshold=0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        self.is_running = False
        
    def start_detection(self, video_source=0):
        """Start real-time detection"""
        self.is_running = True
        
        # Start video capture thread
        video_thread = threading.Thread(target=self._capture_frames, args=(video_source,))
        video_thread.start()
        
        # Start detection thread
        detection_thread = threading.Thread(target=self._process_frames)
        detection_thread.start()
        
        return video_thread, detection_thread
    
    def _capture_frames(self, video_source):
        """Capture frames from video source"""
        cap = cv2.VideoCapture(video_source)
        
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
                
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            else:
                # Skip frame if queue is full
                continue
                
        cap.release()
    
    def _process_frames(self):
        """Process frames for detection"""
        while self.is_running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                
                # Run detection
                results = self.model(frame)[0]
                
                detections = []
                for r in results.boxes.data.tolist():
                    x1, y1, x2, y2, conf, cls = r
                    
                    if conf > self.confidence_threshold:
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': conf,
                            'timestamp': datetime.now().isoformat()
                        }
                        detections.append(detection)
                        
                        # Draw detection on frame
                        cv2.rectangle(frame, 
                                    (int(x1), int(y1)), 
                                    (int(x2), int(y2)), 
                                    (0, 255, 0), 2)
                        
                self.result_queue.put({
                    'frame': frame,
                    'detections': detections
                })
    
    def get_latest_result(self):
        """Get the latest detection result"""
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None
    
    def stop_detection(self):
        """Stop detection threads"""
        self.is_running = False
    
    def log_detection(self, detection):
        """Log detection for analysis"""
        with open('detection_log.json', 'a') as f:
            json.dump(detection, f)
            f.write('\n')

if __name__ == "__main__":
    # Initialize detector
    detector = WasteDetectorDeployment(
        model_path='path/to/trained/model/best.pt',
        confidence_threshold=0.5
    )
    
    # Start detection
    video_thread, detection_thread = detector.start_detection()
    
    try:
        while True:
            # Get latest detection result
            result = detector.get_latest_result()
            if result:
                frame = result['frame']
                detections = result['detections']
                
                # Display frame
                cv2.imshow('Pet Waste Detection', frame)
                
                # Log detections
                for detection in detections:
                    detector.log_detection(detection)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
    finally:
        # Cleanup
        detector.stop_detection()
        video_thread.join()
        detection_thread.join()
        cv2.destroyAllWindows()