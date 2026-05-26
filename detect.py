
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model("https://ultralytics.com/images/bus.jpg")

results[0].save(filename="results.jpg")

print("Detection completed.")
print("Output saved as results.jpg")
