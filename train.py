import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description="Train RoadGuard YOLO pothole detector")
    parser.add_argument("--data", default=str(ROOT / "dataset" / "data.yaml"))
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    data = Path(args.data)
    if not data.exists():
        raise SystemExit("Dataset YAML missing. Follow backend/dataset/README.md.")
    model = YOLO(args.model)
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, project=str(ROOT / "runs"), name="roadguard-pothole")
    candidates = list((ROOT / "runs").glob("**/weights/best.pt"))
    if not candidates:
        raise SystemExit("Training finished but best.pt was not found.")
    destination = ROOT / "models" / "best.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidates[-1], destination)
    print("Saved trained weights to", destination)

if __name__ == "__main__":
    main()
