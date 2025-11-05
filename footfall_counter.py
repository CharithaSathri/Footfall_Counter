"""
Footfall Counter using Computer Vision
--------------------------------------
Counts people entering and exiting through a region (like a doorway) in a video.

Features:
✅ YOLOv8 for real-time person detection (pretrained on COCO)
✅ Simple centroid-based tracking
✅ Line-crossing logic for entry/exit counting
✅ Works on webcam or video input
✅ Displays processed video with counts
"""

# --------------------- IMPORTS ---------------------
from ultralytics import YOLO
import cv2
import numpy as np
import sys

# --------------------- MAIN FUNCTION ---------------------
def main():
    # Load YOLOv8 model (auto-downloads weights if not present)
    model = YOLO("yolov8s.pt")  # You can switch to 'yolov8s.pt' for higher accuracy
    from ultralytics import YOLO
import cv2

model = YOLO('yolov8s.pt')
cap = cv2.VideoCapture('data/test_video.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break   # ✅ properly indented

    frame = cv2.resize(frame, (640, 360))  # downscale for speed

    results = model(frame, conf=0.25)
    annotated = results[0].plot()
    cv2.imshow("YOLO Detection Test", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()

    # --- Video Input ---
    # Use webcam (0) or video file path, e.g., 'data/test_video.mp4'
    cap = cv2.VideoCapture('data/test_video.mp4')

    if not cap.isOpened():
        print("❌ Error: Unable to open video source.")
        sys.exit()

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define counting line (horizontal line)
    line_y = int(frame_height * 0.5)
    line_color = (0, 0, 255)
    line_thickness = 3

    # Initialize variables
    in_count = 0
    out_count = 0
    track_history = {}  # Track centroids per ID

    # Output video writer
    out = cv2.VideoWriter("outputs/result.mp4",
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          30, (frame_width, frame_height))

    print("Processing video... Press 'Q' to quit.")

    # --------------------- PROCESS LOOP ---------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 tracking (detect only 'person' class = 0)
        results = model.track(frame, persist=True, conf=0.25)



        # Draw counting line
        cv2.line(frame, (0, line_y), (frame_width, line_y), line_color, line_thickness)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = box
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # Store centroid history
                if track_id not in track_history:
                    track_history[track_id] = []
                track_history[track_id].append((cx, cy))

                # Keep last 10 points for memory efficiency
                if len(track_history[track_id]) > 10:
                    track_history[track_id].pop(0)

                # Draw bounding box and ID
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                # Counting Logic (crossing the line)
                if len(track_history[track_id]) >= 2:
                    y_positions = [p[1] for p in track_history[track_id][-2:]]
                    if y_positions[-2] < line_y and y_positions[-1] > line_y:
                        in_count += 1
                        cv2.line(frame, (0, line_y), (frame_width, line_y), (0, 255, 0), 5)
                    elif y_positions[-2] > line_y and y_positions[-1] < line_y:
                        out_count += 1
                        cv2.line(frame, (0, line_y), (frame_width, line_y), (255, 0, 0), 5)

        # Display count
        cv2.putText(frame, f"IN: {in_count} | OUT: {out_count}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Show and save frame
        cv2.imshow("Footfall Counter", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --------------------- CLEANUP ---------------------
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\n✅ Processing complete!")
    print(f"People Entered: {in_count}")
    print(f"People Exited: {out_count}")
    print("Output saved as: outputs/result.mp4")

# --------------------- ENTRY POINT ---------------------
if __name__ == "__main__":
    main()
