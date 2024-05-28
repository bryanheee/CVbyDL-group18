import os
import cv2
import numpy as np
from piqe import piqe
import pickle
# from pypiqe import piqe -> same output as as gh code, and it is validated against the matlab code



def save_dict_to_file(dictionary, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as file:
        pickle.dump(dictionary, file)

def count_image_resolutions(image_dir):
    image_files = [f for f in os.listdir(image_dir)]
    for f in image_files:
        image_path = os.path.join(image_dir, f)
        image = cv2.imread(image_path)
        if image is not None:
            if image.shape not in ret:
                ret[image.shape] = []
            ret[image.shape].append(image_path)

def analyze_images(image_paths, target_resolution):
    piqe_scores = []
    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        image_resized = cv2.resize(image, target_resolution)
        piqe_score = piqe(image_resized)[0]
        score = piqe(image_resized)[0]   
        print(f"the score computed by python module {score}")
        # print(f"the score computed by github code {piqe_score}")
        # break
        piqe_scores.append(piqe_score)
        print(f"now done with computing image number {i}, piqe score = {piqe_score}")
    avg_piqe = np.mean(piqe_scores) if piqe_scores else 0
    return piqe_scores

def main():
    base_dir = 'archive/Warp-D'
    train_images_dir = os.path.join(base_dir, 'train/images')
    test_images_dir = os.path.join(base_dir, 'test/images')
    count_image_resolutions(train_images_dir)
    count_image_resolutions(test_images_dir)
    
    print("Image Resolutions:")

    for resolution, count in ret.items():
        print(f"Resolution {resolution}: {len(count)} images")
    # Analysis
    high_res = (1080, 1920, 3)
    low_res = (540, 960, 3)
    
    if high_res in ret:
        print("\nHigh Resolution (1080, 1920) Train Images Analysis:")
        high_res_piqe_scores = analyze_images(ret[high_res], (high_res[0], high_res[1]) )
        print(f"Average PIQE: {np.mean(high_res_piqe_scores) if high_res_piqe_scores else 0:.2f}")
        save_dict_to_file(high_res_piqe_scores, "piqe_scores/high_res_piqe_scores.pkl")
        
        print("\nResized to Low Resolution (540, 960):")
        high_res_to_low_piqe_scores = analyze_images(ret[high_res], (low_res[0], low_res[1]))
        print(f"Average PIQE: {np.mean(high_res_to_low_piqe_scores) if high_res_to_low_piqe_scores else 0:.2f}")
        save_dict_to_file(high_res_to_low_piqe_scores, "piqe_scores/high_res_to_low_piqe_scores.pkl")
    
    
    if low_res in ret:
        print("\nLow Resolution (540, 960) Train Images Analysis:")
        low_res_piqe_scores = analyze_images(ret[low_res], (low_res[0], low_res[1]))
        print(f"Average PIQE: {np.mean(low_res_piqe_scores) if low_res_piqe_scores else 0:.2f}")
        save_dict_to_file(low_res_piqe_scores, "piqe_scores/low_res_piqe_scores.pkl")
        
        print("\nResized to High Resolution (1080, 1920):")
        low_res_to_high_piqe_scores = analyze_images(ret[low_res], (high_res[0], high_res[1]))
        print(f"Average PIQE: {np.mean(low_res_to_high_piqe_scores) if low_res_to_high_piqe_scores else 0:.2f}")
        save_dict_to_file(low_res_to_high_piqe_scores, "piqe_scores/low_res_to_high_piqe_scores.pkl")

if __name__ == "__main__":
    ret = {}
    main()