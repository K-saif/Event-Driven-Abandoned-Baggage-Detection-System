---

# 🧳 Person–Bag Association & Tracking System

This project performs **real-time tracking and reasoning** over persons and their belongings (e.g. backpacks, handbags, suitcases) using:

* **YOLOv11** for object detection
* **DeepSORT** for tracking
* **Custom spatial + temporal logic** for semantic understanding

With this system, you can automatically determine:

✔ Who owns which bag
✔ When a person drops a bag and moves away
✔ When a bag is left behind
✔ When a person leaves without their bag

This is useful for **airport surveillance**, **cafes**, **public spaces**, **security monitoring**, and **smart retail**.

---

## ✨ Features

### 🧠 **Object Detection**

Uses YOLOv11 to detect:

* `person`
* `backpack`
* `handbag`
* `suitcase`

### 🎯 **Object Tracking**

Tracks identities consistently using **DeepSORT**:

* Each person gets a track ID (`P1`, `P2`, …)
* Each bag gets a track ID (`B5`, `B10`, …)

### 🔗 **Person ↔ Bag Association Logic**

Based on:

* **Spatial proximity**
* **Temporal consistency**
* **Track ID reasoning**

### 🚨 **Event Understanding**

System detects events such as:

| Event                         | Description                                |
| ----------------------------- | ------------------------------------------ |
| **Ownership**                 | Person is currently carrying/holding a bag |
| **Separation**                | Person drops bag and moves away            |
| **Bag Left Behind**           | Bag stays in frame without owner           |
| **Person Leaves Without Bag** | Person exits frame leaving bag behind      |

### 🎥 **Output Video Rendering**

Generates a processed video showing:

* Bounding boxes
* Track IDs
* Event alerts
* Ownership confirmations

---

## 🏗️ System Architecture

```
 YOLOv11 → DeepSORT → Spatial Logic → Temporal Logic → Event Reasoning → Output Video
```

---

## 🧩 Tech Stack

| Component        | Library                 |
| ---------------- | ----------------------- |
| Object Detection | `ultralytics (YOLOv11)` |
| Tracking         | `deep-sort-realtime`    |
| Computer Vision  | `OpenCV`                |
| Math Utilities   | `math`, `collections`   |
| Video Processing | `OpenCV VideoWriter`    |

---

## 📦 Installation

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/person-bag-tracking.git
cd person-bag-tracking
```

### 2. Create & activate environment (optional)

```bash
conda create -n tracking python=3.10 -y
conda activate tracking
```

### 3. Install dependencies

```bash
pip install ultralytics deep-sort-realtime opencv-python
```

---

## 🚀 Usage

### 1. Place your input video

Put your source video (e.g. `1.mp4`) in the project directory.

### 2. Run main script

```bash
python main.py
```

### 3. Output

Processed video will be saved as:

```
output2.mp4
```

Alerts and annotations appear on-screen.

---

## 🎛️ Configuration

You can tune thresholds inside `PersonBagTracker`:

| Parameter              | Default | Meaning                                     |
| ---------------------- | ------- | ------------------------------------------- |
| `FRAME_THRESHOLD`      | 8       | Frames required to confirm ownership        |
| `SEPARATION_THRESHOLD` | 200     | Distance threshold (pixels) for separation  |
| `SEPARATION_FRAMES`    | 8       | Frames needed to confirm bag drop           |
| `RELEASE_THRESHOLD`    | 12      | Frames before bag is considered left behind |

Adjust based on:

* Camera angle
* Resolution
* Environment scale

---

## 🧪 Example Events Detected

During execution you might see overlays like:

```
Person 3 MOVED AWAY from Bag 5
Bag 5 LEFT BEHIND
ALERT: Person 3 left WITHOUT Bag 5
```

---

## 📁 Project Structure

```
project/
│
├── main.py                # Main entry point
├── README.md              # Documentation
├── 1.mp4                  # Sample input
└── output2.mp4            # Processed output
```

---

## 🏭 Real-World Applications

* ✈️ Airport baggage safety
* 🏬 Shopping malls & retail stores
* ☕ Cafes & co-working spaces
* 🚆 Train & bus terminals
* 🎒 School and campus monitoring
* 🏢 Corporate building security

---

## 📈 Future Improvements

Possible future extensions include:

* Multi-camera person re-identification
* CLIP-based visual matching
* MQTT / WebSocket event streaming
* Logging to CSV / database
* Real-time dashboard UI
* Edge deployment (Jetson / OAK-D)

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---
