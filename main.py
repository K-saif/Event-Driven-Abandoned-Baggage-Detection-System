from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import math
from collections import defaultdict

def center(bbox):
    l, t, r, b = bbox
    return (int((l + r) / 2), int((t + b) / 2))

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def main():
    model = YOLO("yolo11n.pt")
    tracker = DeepSort(max_age=30, n_init=5, max_iou_distance=0.9, embedder="mobilenet", half=True)

    cap = cv2.VideoCapture("3.mp4")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter("output2.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    # Memory
    association_count = defaultdict(int)
    FRAME_THRESHOLD = 8

    person_owns = defaultdict(set)  # pid → set(bag_ids)
    bag_owners = defaultdict(set)   # bid → set(person_ids)

    bag_release_count = defaultdict(int)
    RELEASE_THRESHOLD = 12

    bag_separation_count = defaultdict(int)
    SEPARATION_THRESHOLD = 200   # pixels
    SEPARATION_FRAMES = 8        # consecutive frames before confirming drop

    frame_id = 0
    last_seen_person_frame = {}
    last_seen_bag_frame = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        results = model(frame, conf=0.4, iou=0.4, classes=[0, 24, 26, 28], verbose=False)

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                if (x2 - x1) < 40 or (y2 - y1) < 80:
                    continue
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))

        detections = sorted(detections, key=lambda x: x[1], reverse=True)
        tracks = tracker.update_tracks(detections, frame=frame)

        person_tracks = {}
        bag_tracks = {}

        # Collect persons & bags
        for track in tracks:
            if not track.is_confirmed():
                continue

            tid = track.track_id
            l, t, r, b = track.to_ltrb()
            cls = track.get_det_class()

            if cls == 0:
                person_tracks[tid] = (l, t, r, b)
                last_seen_person_frame[tid] = frame_id
            elif cls in [24, 26, 28]:
                bag_tracks[tid] = (l, t, r, b)
                last_seen_bag_frame[tid] = frame_id

        # --- Spatial + Temporal Association ---
        for bid, b_box in bag_tracks.items():
            b_center = center(b_box)
            best_pid = None
            best_dist = 99999

            for pid, p_box in person_tracks.items():
                p_center = center(p_box)
                dist = distance(p_center, b_center)
                if dist < best_dist:
                    best_dist = dist
                    best_pid = pid

            if best_pid is not None and best_dist < 120:
                association_count[(best_pid, bid)] += 1

        # --- Confirm Ownership ---
        for (pid, bid), count in association_count.items():
            if count > FRAME_THRESHOLD:
                person_owns[pid].add(bid)
                bag_owners[bid].add(pid)

        # --- Bag Release Logic (no owner nearby OR no owner visible) ---
        for bid in bag_owners:
            if bid not in bag_tracks:
                bag_release_count[bid] += 1
            else:
                visible_owners = any([oid in person_tracks for oid in bag_owners[bid]])
                if not visible_owners:
                    bag_release_count[bid] += 1
                else:
                    bag_release_count[bid] = 0

        # --- Separation Logic (Person moves away but still visible) ---
        for pid, bags in person_owns.items():
            for bid in list(bags):
                if bid in bag_tracks and pid in person_tracks:
                    # both visible
                    p_center = center(person_tracks[pid])
                    b_center = center(bag_tracks[bid])
                    dist = distance(p_center, b_center)

                    if dist > SEPARATION_THRESHOLD:
                        bag_separation_count[(pid, bid)] += 1
                    else:
                        bag_separation_count[(pid, bid)] = 0

                    if bag_separation_count[(pid, bid)] > SEPARATION_FRAMES:
                        cv2.putText(frame,
                            f"Person {pid} MOVED AWAY from Bag {bid}",
                            (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        cv2.putText(frame,
                            f"Bag {bid} LEFT BEHIND",
                            (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                        # Mark bag as released & remove association
                        bag_release_count[bid] = RELEASE_THRESHOLD + 1
                        bags.remove(bid)
                        bag_owners[bid].discard(pid)

        # --- Person Leaves Without Bag ---
        for pid in list(person_owns.keys()):
            if pid not in person_tracks:
                for bid in person_owns[pid]:
                    if bag_release_count[bid] < RELEASE_THRESHOLD:
                        cv2.putText(frame,
                            f"ALERT: Person {pid} left WITHOUT Bag {bid}",
                            (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                del person_owns[pid]

        # --- Draw Tracking Boxes ---
        for track in tracks:
            if not track.is_confirmed():
                continue

            tid = track.track_id
            l, t, r, b = track.to_ltrb()
            cls = track.get_det_class()

            color = (0, 255, 0) if cls == 0 else (255, 0, 0)
            label = f"P{tid}" if cls == 0 else f"B{tid}"

            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), color, 2)
            cv2.putText(frame, label, (int(l), int(t) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # --- Draw Left Behind Bags ---
        for bid, rel_count in bag_release_count.items():
            if rel_count > RELEASE_THRESHOLD and bid in bag_tracks:
                lx1, ly1, lx2, ly2 = bag_tracks[bid]
                cv2.putText(frame,
                    f"Bag {bid} LEFT BEHIND",
                    (int(lx1), int(ly1) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Tracking", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("✔ DONE — output2.mp4 saved with full logic")

if __name__ == "__main__":
    main()
