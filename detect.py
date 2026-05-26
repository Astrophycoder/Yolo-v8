
import sys
from ultralytics import YOLO


def main():
    model_name = "yolov8n.pt"
    source_img = "https://ultralytics.com/images/bus.jpg"
    output_filename = "results.jpg"

    try:
        print(f"Loading model: {model_name}")
        model = YOLO(model_name)

        print(f"Running detection on: {source_img}")
        results = model(source_img)

        results[0].save(filename=output_filename)

        print("\nDetection Results:")

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            print(
                f"Detected: {class_name} | "
                f"Confidence: {confidence:.2f}"
            )

        print(f"\nOutput saved as {output_filename}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
