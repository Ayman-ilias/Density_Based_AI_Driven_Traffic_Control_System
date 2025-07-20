import torch
import cv2
import numpy as np
from collections import defaultdict, deque
import time
import threading
from queue import Queue
import math

# Configuration Variables - Change these as needed
MODEL_PATH = #add trained model path here
VIDEO_PATH =  # Input video path
OUTPUT_PATH =  # Output video path
CONFIDENCE_THRESHOLD = 0.3

class VehicleTracker:
    def __init__(self, max_disappeared=30, max_distance=100):
        """
        Advanced vehicle tracker with smooth bounding boxes
        
        Args:
            max_disappeared (int): Maximum frames a vehicle can disappear before removal
            max_distance (float): Maximum distance for matching vehicles across frames
        """
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # For smooth tracking
        self.position_history = defaultdict(lambda: deque(maxlen=5))
        self.confidence_history = defaultdict(lambda: deque(maxlen=3))
        
    def register(self, centroid, bbox, confidence):
        """Register a new vehicle"""
        self.objects[self.next_id] = {
            'centroid': centroid,
            'bbox': bbox,
            'confidence': confidence,
            'age': 0,
            'stable': False
        }
        self.disappeared[self.next_id] = 0
        self.position_history[self.next_id].append(centroid)
        self.confidence_history[self.next_id].append(confidence)
        self.next_id += 1
        
    def deregister(self, object_id):
        """Remove a vehicle from tracking"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.position_history:
            del self.position_history[object_id]
        if object_id in self.confidence_history:
            del self.confidence_history[object_id]
            
    def update(self, detections):
        """Update tracker with new detections"""
        if len(detections) == 0:
            # Mark all existing objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.get_trackable_objects()
        
        # Initialize arrays for new detections
        input_centroids = []
        input_bboxes = []
        input_confidences = []
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids.append((cx, cy))
            input_bboxes.append(bbox)
            input_confidences.append(det['confidence'])
        
        # If no existing objects, register all new ones
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_bboxes[i], input_confidences[i])
        else:
            # Match existing objects to new detections
            object_centroids = [obj['centroid'] for obj in self.objects.values()]
            object_ids = list(self.objects.keys())
            
            # Compute distance matrix
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - np.array(input_centroids), axis=2)
            
            # Find minimum distances and update objects
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_row_indices = set()
            used_col_indices = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                    
                if D[row, col] <= self.max_distance:
                    object_id = object_ids[row]
                    
                    # Smooth position update
                    old_centroid = self.objects[object_id]['centroid']
                    new_centroid = input_centroids[col]
                    
                    # Apply smoothing
                    smoothed_centroid = self.smooth_position(object_id, new_centroid)
                    smoothed_bbox = self.smooth_bbox(object_id, input_bboxes[col])
                    
                    self.objects[object_id]['centroid'] = smoothed_centroid
                    self.objects[object_id]['bbox'] = smoothed_bbox
                    self.objects[object_id]['confidence'] = input_confidences[col]
                    self.objects[object_id]['age'] += 1
                    
                    # Mark as stable after being tracked for several frames
                    if self.objects[object_id]['age'] > 5:
                        self.objects[object_id]['stable'] = True
                    
                    self.disappeared[object_id] = 0
                    
                    used_row_indices.add(row)
                    used_col_indices.add(col)
            
            # Handle unmatched detections and objects
            unused_row_indices = set(range(0, D.shape[0])) - used_row_indices
            unused_col_indices = set(range(0, D.shape[1])) - used_col_indices
            
            # Mark unmatched objects as disappeared
            for row in unused_row_indices:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Register new objects
            for col in unused_col_indices:
                self.register(input_centroids[col], input_bboxes[col], input_confidences[col])
        
        return self.get_trackable_objects()
    
    def smooth_position(self, object_id, new_centroid):
        """Apply smoothing to position"""
        self.position_history[object_id].append(new_centroid)
        positions = list(self.position_history[object_id])
        
        if len(positions) < 2:
            return new_centroid
        
        # Weighted average with more weight on recent positions
        weights = [0.1, 0.15, 0.25, 0.5][-len(positions):]
        weighted_sum = sum(w * np.array(pos) for w, pos in zip(weights, positions))
        total_weight = sum(weights)
        
        smoothed = weighted_sum / total_weight
        return tuple(map(int, smoothed))
    
    def smooth_bbox(self, object_id, new_bbox):
        """Apply smoothing to bounding box"""
        if object_id not in self.objects:
            return new_bbox
        
        old_bbox = self.objects[object_id]['bbox']
        
        # Smooth bbox coordinates
        alpha = 0.7  # Smoothing factor
        smoothed_bbox = [
            alpha * new_bbox[0] + (1 - alpha) * old_bbox[0],
            alpha * new_bbox[1] + (1 - alpha) * old_bbox[1],
            alpha * new_bbox[2] + (1 - alpha) * old_bbox[2],
            alpha * new_bbox[3] + (1 - alpha) * old_bbox[3]
        ]
        
        return smoothed_bbox
    
    def get_trackable_objects(self):
        """Get all trackable objects"""
        return {obj_id: obj for obj_id, obj in self.objects.items() 
                if obj['stable'] or obj['age'] > 2}

class VideoVehicleDetector:
    def __init__(self, model_path, confidence_threshold=0.5):
        """
        Initialize the video vehicle detector
        
        Args:
            model_path (str): Path to the model file
            confidence_threshold (float): Minimum confidence for detections
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        # Bangladeshi traffic vehicle classes
        self.classes = [
            'bicycle', 'bike', 'bus', 'car', 'cng', 
            'easybike', 'leguna', 'rickshaw', 'truck', 'van', 'ambulance', 'fire truck'
        ]
        
        # Load the model
        self.model = self.load_model()
        
        # Initialize tracker
        self.tracker = VehicleTracker(max_disappeared=20, max_distance=80)
        
        # Performance monitoring
        self.fps_counter = deque(maxlen=30)
        self.detection_times = deque(maxlen=30)
        
    def load_model(self):
        """Load the trained model"""
        try:
            from ultralytics import YOLO
            model = YOLO(self.model_path)
            model.conf = self.confidence_threshold
            print("✓ Model loaded successfully")
            return model
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return None
    
    def detect_frame(self, frame):
        """Detect vehicles in a single frame"""
        if self.model is None:
            return []
        
        try:
            start_time = time.time()
            
            # Run inference
            results = self.model.predict(frame, conf=self.confidence_threshold, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        conf = float(boxes.conf[i])
                        if conf >= self.confidence_threshold:
                            xyxy = boxes.xyxy[i].cpu().numpy()
                            cls = int(boxes.cls[i])
                            
                            detection = {
                                'bbox': xyxy.tolist(),
                                'confidence': conf,
                                'class': self.classes[cls] if cls < len(self.classes) else f'class_{cls}',
                                'class_id': cls
                            }
                            detections.append(detection)
            
            # Record detection time
            detection_time = time.time() - start_time
            self.detection_times.append(detection_time)
            
            return detections
            
        except Exception as e:
            print(f"❌ Detection error: {e}")
            return []
    
    def draw_tracking_info(self, frame, tracked_objects, fps, total_detected):
        """Draw tracking information on frame"""
        height, width = frame.shape[:2]
        
        # Hacker-style bright green color
        hacker_green = (0, 255, 65)  # BGR format for OpenCV
        
        # Draw bounding boxes for tracked objects
        for obj_id, obj in tracked_objects.items():
            bbox = obj['bbox']
            confidence = obj['confidence']
            age = obj['age']
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw thin bright green bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), hacker_green, 1)
            
            # Optional: Draw tracking line for stable objects
            if obj['stable'] and age > 10:
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                cv2.circle(frame, center, 2, hacker_green, -1)
        
        # Draw info panel
        self.draw_info_panel(frame, fps, total_detected, len(tracked_objects))
        
        return frame
    
    def draw_info_panel(self, frame, fps, total_detected, current_tracked):
        """Draw information panel on frame"""
        height, width = frame.shape[:2]
        
        # Colors
        hacker_green = (0, 255, 65)
        panel_bg = (0, 0, 0)
        
        # Panel dimensions
        panel_width = 300
        panel_height = 120
        panel_x = width - panel_width - 10
        panel_y = 10
        
        # Draw semi-transparent panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), panel_bg, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw panel border
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_width, panel_y + panel_height), hacker_green, 2)
        
        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Draw text information
        texts = [
            f"VEHICLE DETECTION SYSTEM",
            f"FPS: {fps:.1f}",
            f"Total Detected: {total_detected}",
            f"Currently Tracked: {current_tracked}",
            f"Avg Detection Time: {np.mean(self.detection_times)*1000:.1f}ms"
        ]
        
        for i, text in enumerate(texts):
            y_pos = panel_y + 25 + (i * 20)
            if i == 0:  # Title
                cv2.putText(frame, text, (panel_x + 10, y_pos), font, 0.5, hacker_green, thickness)
            else:
                cv2.putText(frame, text, (panel_x + 10, y_pos), font, font_scale, hacker_green, thickness)
        
        # Draw real-time graph (simple FPS indicator)
        if len(self.fps_counter) > 1:
            graph_x = panel_x + 10
            graph_y = panel_y + panel_height - 25
            graph_width = panel_width - 20
            graph_height = 15
            
            # Draw graph background
            cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_width, graph_y + graph_height), (40, 40, 40), -1)
            
            # Draw FPS bars
            fps_values = list(self.fps_counter)
            max_fps = max(fps_values) if fps_values else 30
            
            for i, fps_val in enumerate(fps_values[-20:]):  # Show last 20 FPS values
                bar_height = int((fps_val / max_fps) * graph_height)
                bar_x = graph_x + (i * (graph_width // 20))
                bar_y = graph_y + graph_height - bar_height
                
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 2, graph_y + graph_height), hacker_green, -1)
    
    def process_video(self, video_path, output_path):
        """Process video with vehicle detection and tracking"""
        print(f"🎬 Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video Info: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Processing variables
        frame_count = 0
        total_detections = 0
        start_time = time.time()
        
        print("🔄 Starting video processing...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start_time = time.time()
            
            # Detect vehicles in current frame
            detections = self.detect_frame(frame)
            
            # Update tracker
            tracked_objects = self.tracker.update(detections)
            
            # Update statistics
            total_detections = max(total_detections, len(tracked_objects))
            
            # Calculate FPS
            frame_end_time = time.time()
            frame_fps = 1.0 / (frame_end_time - frame_start_time)
            self.fps_counter.append(frame_fps)
            
            # Draw tracking information
            frame = self.draw_tracking_info(frame, tracked_objects, frame_fps, total_detections)
            
            # Write frame
            out.write(frame)
            
            frame_count += 1
            
            # Progress update
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / frame_count) * (total_frames - frame_count)
                print(f"📊 Progress: {progress:.1f}% | Frame: {frame_count}/{total_frames} | ETA: {eta:.1f}s")
        
        # Cleanup
        cap.release()
        out.release()
        
        # Final statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        
        print(f"\n✅ Video processing completed!")
        print(f"📊 Final Statistics:")
        print(f"   • Total frames processed: {frame_count}")
        print(f"   • Total processing time: {total_time:.1f}s")
        print(f"   • Average FPS: {avg_fps:.1f}")
        print(f"   • Peak vehicles detected: {total_detections}")
        print(f"   • Output saved to: {output_path}")

def run_video_detection():
    """Main function to run video vehicle detection"""
    print("🎬 Advanced Video Vehicle Detection System")
    print(f"Model: {MODEL_PATH}")
    print(f"Video: {VIDEO_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Confidence: {CONFIDENCE_THRESHOLD}")
    print("="*60)
    
    # Initialize detector
    detector = VideoVehicleDetector(
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )
    
    # Process video
    detector.process_video(VIDEO_PATH, OUTPUT_PATH)

if __name__ == "__main__":
    run_video_detection()