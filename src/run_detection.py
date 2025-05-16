import cv2
import time
import json
from datetime import datetime
from pathlib import Path
from detector import WasteDetectorDeployment

class DetectionRunner:
    def __init__(self, model_path, output_dir='detection_results'):
        """
        Initialize the detection runner
        Args:
            model_path: Path to the trained model weights
            output_dir: Directory to save detection results
        """
        self.detector = WasteDetectorDeployment(
            model_path=model_path,
            confidence_threshold=0.5
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Performance metrics
        self.frame_times = []
        self.detection_times = []
        self.total_frames = 0
        self.total_detections = 0
        
    def run(self, video_source=0, duration=None):
        """
        Run real-time detection
        Args:
            video_source: Video source (0 for webcam, or video file path)
            duration: Duration to run in seconds (None for indefinite)
        """
        print("Starting detection system...")
        video_thread, detection_thread = self.detector.start_detection(video_source)
        
        start_time = time.time()
        try:
            while True:
                # Check duration if specified
                if duration and (time.time() - start_time) > duration:
                    break
                
                # Get latest detection result
                result = self.detector.get_latest_result()
                if result:
                    frame = result['frame']
                    detections = result['detections']
                    
                    # Update metrics
                    self.total_frames += 1
                    self.total_detections += len(detections)
                    
                    # Draw performance metrics on frame
                    self._draw_metrics(frame)
                    
                    # Display frame
                    cv2.imshow('Pet Waste Detection', frame)
                    
                    # Log detections
                    for detection in detections:
                        self.detector.log_detection(detection)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
        finally:
            # Cleanup
            self.detector.stop_detection()
            video_thread.join()
            detection_thread.join()
            cv2.destroyAllWindows()
            
            # Save performance metrics
            self._save_metrics()
    
    def _draw_metrics(self, frame):
        """Draw performance metrics on frame"""
        fps = self.total_frames / (time.time() - self.start_time) if hasattr(self, 'start_time') else 0
        detections_per_frame = self.total_detections / self.total_frames if self.total_frames > 0 else 0
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Detections/Frame: {detections_per_frame:.1f}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    def _save_metrics(self):
        """Save performance metrics to file"""
        metrics = {
            'total_frames': self.total_frames,
            'total_detections': self.total_detections,
            'average_fps': self.total_frames / (time.time() - self.start_time),
            'detections_per_frame': self.total_detections / self.total_frames if self.total_frames > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        metrics_file = self.output_dir / f'performance_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    # Initialize runner
    runner = DetectionRunner(
        model_path='runs/train/pet_waste_detector/weights/best.pt',
        output_dir='detection_results'
    )
    
    # Run detection (0 for webcam, or provide video file path)
    print("Starting real-time detection...")
    print("Press 'q' to quit")
    runner.run(video_source=0) 