from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2


def main():
    # Load YOLOv11
    model = YOLO("yolo11n.pt")

    # Init DeepSORT with improved settings
    tracker = DeepSort(
        max_age=30,
        n_init=5,
        max_iou_distance=0.9,
        embedder="mobilenet",
        half=True
    )

    cap = cv2.VideoCapture("2.mp4")

    # Video writer setup
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output2.mp4", fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection (with cleaner output)
        results = model(frame, conf=0.4, iou=0.4, classes=[0,24,26,28], verbose=False)

        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                # Ignore small noisy boxes
                if (x2 - x1) < 40 or (y2 - y1) < 80:
                    continue

                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

        # Sort by confidence high->low
        detections = sorted(detections, key=lambda x: x[1], reverse=True)

        # DeepSORT tracking
        tracks = tracker.update_tracks(detections, frame=frame)

        # Draw tracking results
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = track.to_ltrb()

            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (int(l), int(t) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Tracking", frame)
        out.write(frame)  # Save frame to video

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("✔ Saved output video as output.mp4")

if __name__ == "__main__":
    main()
