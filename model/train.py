"""
PPE Kit Detection Model Training Script (Roboflow Integration)
Trains YOLOv8 on the Roboflow Universe Construction Site Safety Dataset:
- Classes: Hardhat, Safety Vest, Mask, Person, NO-Hardhat, NO-Safety Vest, NO-Mask
"""

import argparse
import os
import sys
import glob
from pathlib import Path

def download_roboflow_dataset(api_key: str = None, version: int = 28) -> str:
    """Downloads the official Construction Site Safety dataset from Roboflow Universe."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[ERROR] roboflow package not installed. Run: pip install roboflow")
        sys.exit(1)

    print("📥 Connecting to Roboflow Universe...")
    if not api_key:
        api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        api_key = input("Enter your free Roboflow API Key (from app.roboflow.com): ").strip()

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-universe-projects").project("construction-site-safety")
    print(f"📦 Downloading dataset version {version} in YOLOv8 format...")
    dataset = project.version(version).download("yolov8")
    data_yaml = Path(dataset.location) / "data.yaml"
    print(f"✅ Roboflow dataset downloaded to: {data_yaml}")
    return str(data_yaml)

def train(
    data_path: str = "data.yaml",
    model_name: str = "yolov8s.pt",
    epochs: int = 40,
    imgsz: int = 640,
    batch_size: int = 32,
    device: str = "0",
    project: str = "ppe_runs",
    name: str = "ppe_roboflow_yolov8s"
):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)

    print("=" * 65)
    print("🚀 Starting PPE Kit Detection Training (Roboflow Dataset)")
    print(f"Base Model:  {model_name}")
    print(f"Data Config: {data_path}")
    print(f"Epochs:      {epochs}")
    print(f"Batch Size:  {batch_size} (Optimized for GPU VRAM)")
    print(f"Cache:       True (RAM acceleration)")
    print(f"Device:      {device}")
    print("=" * 65)

    model = YOLO(model_name)

    # Train model with hardware-optimized parameters
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        patience=12,
        save=True,
        save_period=5,
        plots=True,
        cache=True,
        workers=4,
        verbose=True,
        # Augmentations for robust construction site detection
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4
    )

    print("\n✅ Training complete!")
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path(project) / name
    best_weights = save_dir / "weights" / "best.pt"
    
    if not best_weights.exists():
        found = glob.glob(f"**/{name}/**/best.pt", recursive=True)
        best_weights = Path(found[-1]) if found else best_weights

    print(f"🏆 Best model weights saved to: {best_weights.resolve()}")

    # Run validation
    print("\n📊 Running validation on test/val split...")
    metrics = model.val()
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")

    # Export to ONNX
    print("\n📦 Exporting best model to ONNX format...")
    try:
        model.export(format="onnx")
        print("✅ ONNX model exported successfully!")
    except Exception as e:
        print(f"ONNX export notice: {e}")

    return best_weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPE Kit Detection with Roboflow")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--roboflow", action="store_true", help="Download official dataset from Roboflow Universe")
    parser.add_argument("--api-key", type=str, default=None, help="Roboflow API key")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="YOLO checkpoint (yolov8s.pt, yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", type=str, default="0", help="GPU device id ('0') or 'cpu'")
    parser.add_argument("--project", type=str, default="ppe_runs", help="Project output folder")
    parser.add_argument("--name", type=str, default="ppe_roboflow_yolov8s", help="Run experiment name")

    args = parser.parse_args()

    data_yaml = args.data
    if args.roboflow or not os.path.exists(data_yaml):
        data_yaml = download_roboflow_dataset(api_key=args.api_key)

    train(
        data_path=data_yaml,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        project=args.project,
        name=args.name
    )
