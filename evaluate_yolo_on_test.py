from ultralytics import YOLO
from pathlib import Path


# MAKE SURE TO put the path to the last.py or best.pt
path_to_mixup_yolo = 'runs-low-res-mixup/detect/train/weights/last.pt'
path_to_yolo_no_augmentation = 'runs-low-res-no-augment/detect/train/weights/last.pt'
model = YOLO(path_to_mixup_yolo)

# First swap the test an val path in wapd.yaml!!! or just replace
rel_path_to_mixup_yaml = "low_res_datasets_mixup_lam05000386523859661/warpd.yaml"
rel_path = "DATASETS_LOW_RES/warpd.yaml"
full_path = Path(rel_path_to_mixup_yaml).resolve()


save_test_results_at = 'test/low_res/'
save_test_results_at_mixup = 'test/mixup'
name = 'results'
try:
    val_results = model.val(data=str(full_path), plots=True, save_json = True,  project=save_test_results_at_mixup, name=name)

    print(val_results.box.map)  # map50-95
    print(val_results.box.map50)  # map50
    print(val_results.box.map75)  # map75
    print(val_results.box.maps)  # a list contains map50-95 of each category

except RuntimeError as e:
    print(f"This shouldn't print {e}")
    print(f"================== this is the full path {full_path} ==================")