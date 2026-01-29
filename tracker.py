from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import math
from collections import defaultdict
from datetime import datetime
from utils import log_event


# ===============================
# Utility Functions
# ===============================

def center(bbox):
    l, t, r, b = bbox
    return (int((l + r) / 2), int((t + b) / 2))


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# ===============================
# Person–Bag Tracker Logic Class
# ===============================

class PersonBagTracker:
    def __init__(self, CSV_PATH, IMG_DIR):
        self.association_count = defaultdict(int)
        self.person_owns = defaultdict(set)
        self.bag_owners = defaultdict(set)

        self.logged_events = set()
        self.finalized_bags = set()

        self.bag_release_count = defaultdict(int)
        self.bag_separation_count = defaultdict(int)

        self.FRAME_THRESHOLD = 8
        self.RELEASE_THRESHOLD = 12
        self.SEPARATION_THRESHOLD = 200
        self.SEPARATION_FRAMES = 8

        self.CSV_PATH = CSV_PATH
        self.IMG_DIR = IMG_DIR

    def associate(self, person_tracks, bag_tracks):
        for bid, b_box in bag_tracks.items():
        
            # ✅ CRITICAL LINE
            if bid in self.finalized_bags:
                continue   # NEVER reassign

            b_center = center(b_box)
            best_pid, best_dist = None, float("inf")

            for pid, p_box in person_tracks.items():
                dist = distance(center(p_box), b_center)
                if dist < best_dist:
                    best_dist, best_pid = dist, pid

            if best_pid is not None and best_dist < 120:
                self.association_count[(best_pid, bid)] += 1

    def confirm_ownership(self):
        for (pid, bid), count in list(self.association_count.items()):
            
            if bid in self.finalized_bags:
                continue

            if count > self.FRAME_THRESHOLD:
                self.person_owns[pid].add(bid)
                self.bag_owners[bid].add(pid)

    def check_bag_release(self, person_tracks, bag_tracks):
        for bid in list(self.bag_owners.keys()):
            if bid not in bag_tracks:
                self.bag_release_count[bid] += 1
            else:
                active_owner_visible = any(pid in person_tracks for pid in self.bag_owners[bid])
                self.bag_release_count[bid] += 0 if active_owner_visible else 1

    def check_separation(self, person_tracks, bag_tracks, frame, frame_number):
        for pid, bags in self.person_owns.items():

            for bid in list(bags):
                if bid in self.finalized_bags:
                    continue
                if bid in bag_tracks and pid in person_tracks:
                    dist = distance(center(person_tracks[pid]), center(bag_tracks[bid]))

                    if dist > self.SEPARATION_THRESHOLD:
                        self.bag_separation_count[(pid, bid)] += 1
                    else:
                        self.bag_separation_count[(pid, bid)] = 0

                    if self.bag_separation_count[(pid, bid)] > self.SEPARATION_FRAMES:
                        self._handle_separation(frame, pid, bid, frame_number)
                        self.person_owns[pid].remove(bid)
                        self.bag_owners[bid].discard(pid)
                        self.bag_release_count[bid] = self.RELEASE_THRESHOLD + 1

    def check_person_exit(self, person_tracks, frame, frame_number):
        for pid in list(self.person_owns.keys()):
            if pid not in person_tracks:
                for bid in self.person_owns[pid]:
                    if self.bag_release_count[bid] < self.RELEASE_THRESHOLD:
                        self._handle_exit(frame, pid, bid, frame_number)
                del self.person_owns[pid]

    # ===============================
    # Draw & Trigger Methods
    # ===============================

    def draw_boxes(self, frame, tracks):
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

    def draw_left_behind(self, frame, bag_tracks):
        for bid, count in self.bag_release_count.items():
            if count > self.RELEASE_THRESHOLD and bid in bag_tracks:
                l, t, _, _ = bag_tracks[bid]
                cv2.putText(frame, f"Bag {bid} LEFT BEHIND",
                            (int(l), int(t) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 255), 2)

    # def _handle_separation(self, frame, pid, bid, frame_number):
    #     cv2.putText(frame, f"Person {pid} MOVED AWAY from Bag {bid}",
    #                 (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    #     cv2.putText(frame, f"Bag {bid} LEFT BEHIND",
    #                 (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    #     log_event(frame, frame_number, "MOVED_AWAY", pid, bid, self.CSV_PATH, self.IMG_DIR)
    #     log_event(frame, frame_number, "LEFT_BEHIND", pid, bid, self.CSV_PATH, self.IMG_DIR)

    # def _handle_exit(self, frame, pid, bid, frame_number):
    #     cv2.putText(frame, f"ALERT: Person {pid} LEFT WITHOUT Bag {bid}",
    #                 (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    #     log_event(frame, frame_number, "LEFT_WITHOUT_BAG", pid, bid, self.CSV_PATH, self.IMG_DIR)


    def _handle_separation(self, frame, pid, bid, frame_number):

        moved_key = (pid, bid, "MOVED_AWAY")
        left_key = (pid, bid, "LEFT_BEHIND")

        # Log only once
        if moved_key not in self.logged_events:
            cv2.putText(frame, f"Person {pid} MOVED AWAY from Bag {bid}",
                        (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            log_event(frame, frame_number, "MOVED_AWAY", pid, bid, self.CSV_PATH, self.IMG_DIR)
            self.logged_events.add(moved_key)

            # ⭐ FREEZE BAG HERE
            self.finalized_bags.add(bid)

        if left_key not in self.logged_events:
            cv2.putText(frame, f"Bag {bid} LEFT BEHIND",
                        (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            log_event(frame, frame_number, "LEFT_BEHIND", pid, bid, self.CSV_PATH, self.IMG_DIR)
            self.logged_events.add(left_key)


    def _handle_exit(self, frame, pid, bid, frame_number):

        exit_key = (pid, bid, "LEFT_WITHOUT_BAG")

        if exit_key in self.logged_events:
            return

        cv2.putText(frame, f"ALERT: Person {pid} LEFT WITHOUT Bag {bid}",
                    (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        log_event(frame, frame_number, "LEFT_WITHOUT_BAG", pid, bid, self.CSV_PATH, self.IMG_DIR)
        self.logged_events.add(exit_key)
