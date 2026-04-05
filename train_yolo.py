from ultralytics import YOLO
import os

PROJECT_ROOT = "./yolo_particle"
DATA_ROOT = f"{PROJECT_ROOT}/dataset"
RUN_ROOT  = f"{PROJECT_ROOT}/runs"

# Dataset YAML
yaml_path = f"{DATA_ROOT}/dataset.yaml"
with open(yaml_path,"w") as f:
    f.write(f"""
path: {DATA_ROOT}
train: images/train
val: images/val

names:
  0: Fa
  1: Ne
""")

# Train YOLO
model = YOLO("yolov8n.pt")
model.train(
    data=yaml_path,
    epochs=10,      # reduce for demo
    batch=4,
    imgsz=512,
    save_period=5,
    project=RUN_ROOT,
    name="run1",
    device=0,
    workers=2,
    verbose=True
)
