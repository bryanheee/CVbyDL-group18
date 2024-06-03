import argparse
import os
import shutil
from ultralytics import YOLO
from pathlib import Path

def train_yolo_model(rel_path, epochs=50, save_dir="runs"):
    model = YOLO("yolov8n.pt")
    full_path = Path(rel_path).resolve()
    try:
        # Train the model
        results = model.train(data=str(full_path), epochs=epochs, plots=True)
        
        # # Export the model in different formats
        # success_torchscript = model.export(format="torchscript")
        # success = model.export(format="onnx")
        
        # Check if the save_dir already exists and remove it if it does
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        
        # Rename the 'runs' directory to the specified save_dir
        shutil.move("runs", save_dir)
    except RuntimeError as e:
        print(f"This shouldn't print: {e}")
        print(f"================== this is the full path {full_path} ==================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model with specified dataset path.")
    parser.add_argument('--path', type=str, required=True, help='Relative path to the dataset configuration file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train the model')
    parser.add_argument('--save_dir', type=str, default="runs", help='Directory to save the training results')
    
    args = parser.parse_args()
    
    train_yolo_model(args.path, args.epochs, args.save_dir)
