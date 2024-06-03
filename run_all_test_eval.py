import argparse
import os
import subprocess

def main():
    parser = argparse.ArgumentParser(description="Run YOLO model validation multiple times with different arguments")
    parser.add_argument('--yaml_test_path', type=str, required=True, help="Path to the dataset configuration YAML file")
    args = parser.parse_args()
    args.yaml_test_path

    low_res_pt = "yolo_weights/last-original-low-res.pt"
    mixup_pt = "yolo_weights/last-mixup.pt"
    cutout_pt = "yolo_weights/last-cutout.pt"
    cutmix_pt = "yolo_weights/last-cutmix.pt"
    random_pt = "yolo_weights/last-random-aug.pt"
    bernoulli_pt = "yolo_weights/last-bernoulli.pt"

    # Define the different sets of arguments
    model_paths = [low_res_pt, mixup_pt, cutmix_pt, cutout_pt, random_pt, bernoulli_pt]
    save_paths = ["test/low_res/", "test/mixup/", "test/cutmix/", "test/cutout/", "test/random/", "test/bernoulli/"]
    save_paths_aug_test = ["test_aug/low_res/", "test_aug/mixup/", "test_aug/cutmix/", "test_aug/cutout/", "test_aug/random/", "test_aug/bernoulli/"]

    for model_path, save_path_aug in zip(model_paths, save_paths_aug_test):
        output_file = f'{save_path_aug}mAP_table_{os.path.basename(model_path)}.txt'
        os.makedirs(save_path_aug, exist_ok=True)
        with open(output_file, 'w') as f:
            subprocess.run([
                "python", "evaluate_yolo_on_test.py",
                "--model_path", model_path,
                "--yaml_path", args.yaml_test_path,
                "--save_results_path", save_path_aug,
                "--name", "results"
            ], stdout=f, stderr=subprocess.STDOUT)

if __name__ == "__main__":
    main()
