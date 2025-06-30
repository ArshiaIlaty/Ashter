import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import threading
from queue import Queue
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('detector.log'),
        logging.StreamHandler()
    ]
)

class RaspberryPiWasteDetector:
    def __init__(self, model_path='model.tflite', input_size=(640, 640)):
        self.input_size = input_size
        self.confidence_threshold = 0.5
        
        # Initialize TFLite interpreter
        try:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Initialize camera
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not self.cap.isOpened():
                raise Exception("Failed to open camera")
            logging.info("Camera initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing camera: {str(e)}")
            raise
        
        # Threading setup
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        self.running = False
        
        # Performance metrics
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        
    def preprocess_image(self, frame):
        try:
            # Resize and normalize
            img = cv2.resize(frame, self.input_size)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)
            return img
        except Exception as e:
            logging.error(f"Error preprocessing image: {str(e)}")
            return None
        
    def detect_waste(self, frame):
        try:
            # Preprocess image
            input_data = self.preprocess_image(frame)
            if input_data is None:
                return [], [], []
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output tensor
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            # Process detections
            boxes, scores, classes = self.process_output(output_data[0], frame.shape[:2])
            return boxes, scores, classes
        except Exception as e:
            logging.error(f"Error during detection: {str(e)}")
            return [], [], []
        
    def process_output(self, output, original_shape):
        try:
            # Process YOLO output format
            boxes = []
            scores = []
            classes = []
            
            # Get the output tensor shape
            output_shape = output.shape
            
            # Process the output tensor
            for i in range(output_shape[0]):
                confidence = output[i, 4]
                if confidence > self.confidence_threshold:
                    # Get box coordinates
                    x1 = output[i, 0] * original_shape[1]
                    y1 = output[i, 1] * original_shape[0]
                    x2 = output[i, 2] * original_shape[1]
                    y2 = output[i, 3] * original_shape[0]
                    
                    boxes.append([x1, y1, x2, y2])
                    scores.append(confidence)
                    classes.append(0)  # Assuming single class
            
            return boxes, scores, classes
        except Exception as e:
            logging.error(f"Error processing output: {str(e)}")
            return [], [], []
        
    def camera_thread(self):
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame)
                        self.frame_count += 1
                        
                        # Calculate FPS
                        if self.frame_count % 30 == 0:
                            self.fps = self.frame_count / (time.time() - self.start_time)
                            logging.info(f"Current FPS: {self.fps:.2f}")
            except Exception as e:
                logging.error(f"Error in camera thread: {str(e)}")
                    
    def detection_thread(self):
        while self.running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    detections = self.detect_waste(frame)
                    if not self.result_queue.full():
                        self.result_queue.put((frame, detections))
            except Exception as e:
                logging.error(f"Error in detection thread: {str(e)}")
                    
    def start(self):
        self.running = True
        self.start_time = time.time()
        self.frame_count = 0
        
        # Start threads
        threading.Thread(target=self.camera_thread, daemon=True).start()
        threading.Thread(target=self.detection_thread, daemon=True).start()
        logging.info("Detection system started")
        
    def stop(self):
        self.running = False
        self.cap.release()
        logging.info("Detection system stopped")
        
    def get_latest_detection(self):
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None, None

def main():
    try:
        # Create output directory for saved images
        os.makedirs('detections', exist_ok=True)
        
        detector = RaspberryPiWasteDetector()
        detector.start()
        
        while True:
            frame, detections = detector.get_latest_detection()
            if frame is not None and detections is not None:
                boxes, scores, classes = detections
                
                # Draw detections
                for box, score, class_id in zip(boxes, scores, classes):
                    if score > detector.confidence_threshold:
                        x1, y1, x2, y2 = box
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f'Score: {score:.2f}', (int(x1), int(y1)-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw FPS
                cv2.putText(frame, f'FPS: {detector.fps:.1f}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Waste Detection', frame)
                
                # Save frame if detection found
                if len(boxes) > 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f'detections/detection_{timestamp}.jpg', frame)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        logging.error(f"Error in main loop: {str(e)}")
    finally:
        detector.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 