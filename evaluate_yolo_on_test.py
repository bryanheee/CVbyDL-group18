
import argparse
import os
from ultralytics import YOLO
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="YOLO Model Validation Script")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the YOLO model (.pt file)")
    parser.add_argument('--yaml_path', type=str, required=True, help="Path to the dataset configuration YAML file")
    parser.add_argument('--save_results_path', type=str, required=True, help="Directory to save validation results")
    parser.add_argument('--name', type=str, default='results', help="Name for the results")
    
    args = parser.parse_args()

    results_folder_path = os.path.join(args.save_results_path, args.name)
    os.makedirs(results_folder_path, exist_ok=True)

    model = YOLO(args.model_path)
    full_path = Path(args.yaml_path).resolve()

    try:
        val_results = model.val(data=str(full_path), plots=True, save_json=True, project=args.save_results_path, name=args.name)
    except RuntimeError as e:
        print(f"This shouldn't print {e}")
        print(f"================== this is the full path {full_path} ==================")

if __name__ == "__main__":
    main()
