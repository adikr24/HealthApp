import cv2
import numpy as np
from ultralytics import YOLO
import os

class MilkPourVisualizerCustom:
    def __init__(self, video_path, output_video_path, yolo_model, flow_threshold=2.0, white_thresh=200, cup_real_height_cm=None):
        """
        Detect milk pouring using custom YOLO + optical flow focusing on white pixels.
        Optionally estimate volume if cup height is provided.
        """
        self.video_path = video_path
        self.output_video_path = output_video_path
        self.flow_threshold = flow_threshold
        self.white_thresh = white_thresh
        self.cup_real_height_cm = cup_real_height_cm  # Real-world height of the cup in cm (optional)

        # Load YOLO model (default or custom)
        self.model = YOLO(yolo_model)
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    def visualize_pour(self):
        cap = cv2.VideoCapture(self.video_path)
        ret, prev_frame = cap.read()
        if not ret:
            print("Video could not be read.")
            return

        height, width = prev_frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width, height))

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        frame_count = 0
        pouring_frames = 0
        total_pixels_flowed = 0
        cup_pixels = None  # will store total cup pixel count for normalization

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                                None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

            # YOLO detection
            results = self.model(frame)
            mask = np.zeros_like(mag, dtype=np.uint8)

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, cls_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    if cls_name.lower() in ["cup", "milk gallon", "milk pour", "instant coffee"]:
                        roi = frame[y1:y2, x1:x2]
                        # Mask bright/white pixels
                        white_mask = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        white_mask = cv2.threshold(white_mask, self.white_thresh, 1, cv2.THRESH_BINARY)[1]
                        mask[y1:y2, x1:x2] = white_mask

                        # Store total cup pixels for normalization
                        if cup_pixels is None:
                            cup_pixels = np.sum(white_mask)  # approximate area of cup in pixels

            # Detect pouring using masked optical flow
            masked_mag = mag * mask
            flowing_pixels = np.sum(masked_mag > self.flow_threshold)
            total_pixels_flowed += flowing_pixels

            if flowing_pixels > 0:
                pouring_frames += 1
                # Draw arrows only for moving white pixels
                step = 8
                for y in range(0, height, step):
                    for x in range(0, width, step):
                        if mask[y, x] == 1 and np.sqrt(flow[y,x,0]**2 + flow[y,x,1]**2) > self.flow_threshold:
                            dx, dy = flow[y, x]
                            cv2.arrowedLine(frame, (x, y), (int(x+dx), int(y+dy)), (0,0,255), 1, tipLength=0.3)

            out.write(frame)
            prev_gray = gray.copy()
            frame_count += 1

        cap.release()
        out.release()

        print(f"Annotated video saved to {self.output_video_path}")
        print(f"Frames with milk/white pixel pouring detected: {pouring_frames}/{frame_count}")

        if self.cup_real_height_cm and cup_pixels:
            # Estimate volume in cm^3 (simplified)
            # Assume cup diameter ~ height for rough estimate
            cup_area_fraction = total_pixels_flowed / cup_pixels
            estimated_volume_cm3 = cup_area_fraction * self.cup_real_height_cm**3
            print(f"Approximate poured volume: {estimated_volume_cm3:.2f} cm³")
        else:
            print(f"Total flowing pixels detected (relative measure): {total_pixels_flowed}")

# --- Usage ---
input_video_path = "/app/mediaFiles/videos/labeledvideos/pouringMilkandsugar_trimmed.mp4"
output_video_path = "/app/mediaFiles/output/videoOutputs/pouringMilkandsugar/PouringAnnotated_Custom_Volume.mp4"
yolo_model = "yolov8n.pt"

visualizer = MilkPourVisualizerCustom(input_video_path, output_video_path, yolo_model,
                                      flow_threshold=2.0, white_thresh=200,
                                      cup_real_height_cm=10)  # provide real cup height if known
visualizer.visualize_pour()
