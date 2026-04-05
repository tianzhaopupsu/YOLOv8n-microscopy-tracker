"""
YOLOv8 Particle Tracking Visualization
Annotates a video with trained YOLOv8 weights.
"""

import cv2
import torch
from ultralytics import YOLO
import argparse
from pathlib import Path

def visualize_video(video_path, weights_path, output_path, device=None, conf=0.25):
    """
    Annotate a video with YOLOv8 predictions.
    """
    device = device or (0 if torch.cuda.is_available() else "cpu")
    video_path = Path(video_path)
    output_path = Path(output_path)
    
    # Load YOLO model
    model = YOLO(weights_path)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # Video writer
    out = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    frame_count = 0
    print(f"Processing {video_path.name} ...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference
        results = model(frame, device=device, verbose=False, conf=conf)

        # Annotate frame
        annotated_frame = results[0].plot(conf=False, line_width=1, font_size=0.6)

        # Save frame to output video
        out.write(annotated_frame)

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()
    out.release()
    print(f"Saved annotated video to: {output_path}")

# --------------------------
# Command-line interface
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Particle Tracking Visualization")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--weights", required=True, help="Path to YOLOv8 weights")
    parser.add_argument("--output", default="annotated_video.mp4", help="Path to save annotated video")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for YOLO predictions")
    parser.add_argument("--device", default=None, help="Device: 0 for GPU, 'cpu' for CPU")

    args = parser.parse_args()
    visualize_video(args.video, args.weights, args.output, device=args.device, conf=args.conf)
