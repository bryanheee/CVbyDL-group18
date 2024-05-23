import os
import cv2
# from collections import defaultdict

def count_image_resolutions(image_dir):
    # resolution_counts = defaultdict(int)
    ret = {}

    image_files = [f for f in os.listdir(image_dir)]
    for f in image_files:
        image_path = os.path.join(image_dir, f)
        # print(image_path)
        image = cv2.imread(image_path)
        # print(image.shape)
        if image.shape not in ret:
            ret[image.shape] = 1
        else:
            ret[image.shape] = ret[image.shape] + 1
    return ret

def main():
    base_dir = 'archive/Warp-D'
    train_images_dir = os.path.join(base_dir, 'train/images')
    test_images_dir = os.path.join(base_dir, 'test/images')
    
    train_resolutions = count_image_resolutions(train_images_dir)
    test_resolutions = count_image_resolutions(test_images_dir)
    
    print("Train Image Resolutions:")
    for resolution, count in train_resolutions.items():
        print(f"Resolution {resolution}: {count} images")
    
    print("\nTest Image Resolutions:")
    for resolution, count in test_resolutions.items():
        print(f"Resolution {resolution}: {count} images")

if __name__ == "__main__":
    main()