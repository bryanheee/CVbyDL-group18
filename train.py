from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolov8n.pt")
# model = YOLO(checkpoint_path)
rel_path = "DATASETS_LOW_RES/warpd.yaml"



full_path = Path(rel_path).resolve()
try:
    results = model.train(data=str(full_path), epochs=50, plots=True)
    val_results = model.val()
    test_results = model.val(split='test')
    success_torchscript = model.export(format="torchscript")
    success = model.export(format="onnx")
    # print(f"validation _Results = {val_results}")
    # print(f"test_Results = {test_results}")
    # with open("val_results.json", "w") as f:
    #     json.dump(val_results, f)
    # # Save test results
    # with open("test_results.json", "w") as f:
    #     json.dump(test_results, f)   
except RuntimeError as e:
    print(f"This shouldn't print {e}")
    print(f"================== this is the full path {full_path} ==================")