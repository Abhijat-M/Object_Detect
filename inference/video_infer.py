import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import time
import torch
from torchvision import transforms

from models.frcnn import build_model


CHECKPOINT_PATH = "checkpoints/last.pth"
CLASSES = ["__bg__", "person", "car", "dog", "bicycle"]
SCORE_THRESHOLD = 0.3


device = "cuda" if torch.cuda.is_available() else "cpu"

model = build_model(num_classes=len(CLASSES))
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.ToTensor()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

frame_count = 0
fps = 0.0  
start_time = time.time()

print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)[0]

    # FPS calculation (averaged)
    if frame_count % 10 == 0:
        fps = frame_count / (time.time() - start_time)

    # Draw detections
    for box, label, score in zip(
        outputs["boxes"],
        outputs["labels"],
        outputs["scores"]
    ):
        if score < SCORE_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.tolist())
        class_name = CLASSES[label]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{class_name} {score:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # Draw FPS safely
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow("Faster R-CNN (From Scratch)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
