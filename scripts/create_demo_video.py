from __future__ import annotations

import argparse
from pathlib import Path


def create_demo_video(output: str | Path, frame_count: int = 48, fps: float = 12.0) -> Path:
    try:
        import cv2
        import numpy as np
    except Exception as exc:
        raise RuntimeError("OpenCV and NumPy are required to create the demo video") from exc

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 360))
    if not writer.isOpened():
        raise ValueError(f"Unable to create video file: {output_path}")

    try:
        for index in range(frame_count):
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            frame[:, :] = (45 + index % 80, 95, 150)
            x = 40 + (index * 9) % 520
            cv2.rectangle(frame, (x, 120), (x + 80, 210), (230, 230, 230), -1)
            cv2.circle(frame, (x + 40, 250), 24, (20, 20, 20), -1)
            cv2.putText(frame, f"demo frame {index}", (32, 52), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            writer.write(frame)
    finally:
        writer.release()

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small local demo video for keyframe processing.")
    parser.add_argument("--output", default="data/raw/videos/demo_video.mp4")
    parser.add_argument("--frame-count", type=int, default=48)
    parser.add_argument("--fps", type=float, default=12.0)
    args = parser.parse_args()

    path = create_demo_video(args.output, frame_count=args.frame_count, fps=args.fps)
    print(f"Wrote demo video: {path}")


if __name__ == "__main__":
    main()
