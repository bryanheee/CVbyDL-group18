from ultralytics import YOLO
from pathlib import Path
import json

model = YOLO("yolov8n.pt")
rel_path = "datasets/warpd.yaml"
# assert Path(rel_path).exists(), "File doesn't exists"

# results = model.train(data=str(rel_path), epochs=1)
# val_results = model.val()
# test_results = model.val(split='test')

# try:

# except RuntimeError as e:
#     print(
#         f"Using a relative path to the YAML loads the YAML correctly, but "
#         f"garbage gets prepended to the directories specified within the YAML, "
#         f"leading to the following runtime error: \n {e}"
#     )
#     print(rel_path)


full_path = Path(rel_path).resolve()
try:
    results = model.train(data=str(full_path), epochs=50)
    val_results = model.val()
    test_results = model.val(split='test')
    success_torchscript = model.export(format="torchscript")
    success = model.export(format="onnx")
    with open("val_results.json", "w") as f:
        json.dump(val_results, f)
    # Save test results
    with open("test_results.json", "w") as f:
        json.dump(test_results, f)   
except RuntimeError as e:
    print(f"This shouldn't print {e}")
    print(f"================== this is the full path {full_path} ==================")


