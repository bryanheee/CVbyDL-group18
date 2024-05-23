import os
import random
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_image(image, labels=None, text=""):
    """
    Visualizes a normalized image with optional text at the bottom and bounding boxes.
    
    Parameters:
    - image (numpy.ndarray): The normalized image to visualize.
    - labels (numpy.ndarray): The labels for the bounding boxes in YOLO format.
    - text (str): The text to display at the bottom of the image.
    """
    # Denormalize the image if it's normalized
    denormalized_image = (image * 255.0).astype(np.uint8)

    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=(10, 10))

    # Display the image
    ax.imshow(denormalized_image)
    ax.axis('off')  # Hide axes

    # Add bounding boxes if labels are provided
    if labels is not None:
        height, width, _ = denormalized_image.shape
        for label in labels:
            class_id, x_center, y_center, box_width, box_height = label
            # Convert from YOLO format to bounding box format
            x = (x_center - box_width / 2) * width
            y = (y_center - box_height / 2) * height
            w = box_width * width
            h = box_height * height
            # Create a rectangle patch
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
            # Add the rectangle to the plot
            ax.add_patch(rect)
            # Add class label text above the rectangle
            ax.text(x, y, str(int(class_id)), color='red', fontsize=12, verticalalignment='top')

    # Add text at the bottom
    if text:
        plt.text(0.5, -0.05, text, ha='center', va='top', transform=ax.transAxes, fontsize=12)

    plt.show()

def mixup2images(file1, file2, alpha):
    image = cv2.imread(os.path.join(dataset_path, file1), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0  # Normalize

    label_file = file1.replace('.jpg', '.txt').replace('.png', '.txt')
    file = os.path.join(labels_path, label_file)
    arr1 = np.loadtxt(file).reshape(-1, 5)
    # print(file)
    # print(type(arr1))
    # print(arr1)
    # visualize_image(image, arr1 ,f"image1, shape = {image.shape}")

    image2 = cv2.imread(os.path.join(dataset_path, file2), cv2.IMREAD_COLOR)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB).astype(np.float32)
    image2 /= 255.0  # Normalize
    label_file2 = file2.replace('.jpg', '.txt').replace('.png', '.txt')
    file2 = os.path.join(labels_path, label_file2)
    arr2 = np.loadtxt(file2).reshape(-1, 5)
    # print(f"array 2 is \n {arr2}")
    # visualize_image(image2, arr2, f"image2, shape = {image2.shape}")


    # print(f"shape of file 1 is = {image.shape}")
    # print(f"shape of file 2 is = {image2.shape}")

    # resize 'image2' to the dimensions of 'image'
    # print(f"new shape of file 2 = {image2.shape}")  
    # visualize_image(image2, f"image2, reshaped = {image2.shape}")
    # arr2 = adjust_labels(arr2, original_size2, original_size1)
    # print(f"adjusted array 2 is \n {arr2}")

    # print(type(arr2))
    result = np.concatenate((arr1, arr2), axis=0)
    # print(f"result is = \n {result}")
    mixedup_images = lam*image + (1 - lam)*image2
    # visualize_image(mixedup_images, result, f"mixed-up image, shape = {mixedup_images.shape}")

    mixedup_images = (mixedup_images * 255.0).astype(np.uint8) #denormalize
    output_file_path = os.path.join(new_train_path, file1)
    cv2.imwrite(output_file_path, cv2.cvtColor(mixedup_images, cv2.COLOR_RGB2BGR))


    fmt = ['%d'] + ['%.6f'] * (result.shape[1] - 1)
    np.savetxt(os.path.join(new_train_labels_path, label_file), result, delimiter=' ', fmt=fmt, newline='\n')


if __name__ == "__main__":
    
    trainORval = 'train'

    alpha = 1.0
    random.seed(42)
    np.random.seed(42)
    lam = np.random.beta(alpha, alpha)
    lam = np.clip(lam, 0.4, 0.6)
    print(f"lambda is {lam}")

    # Paths
    dataset_path = 'DATASETS_LOW_RES/'+ trainORval +'/images'
    labels_path = 'DATASETS_LOW_RES/'+ trainORval +'/labels'

    new_train_path = 'low_res_datasets_mixup_lam'+ str(lam).replace('.', '') +'/'+ trainORval +'/images'
    new_train_labels_path = 'low_res_datasets_mixup_lam'+ str(lam).replace('.', '') +'/' + trainORval + '/labels'

    # Create directories for new train and val sets
    os.makedirs(new_train_path, exist_ok=True)
    os.makedirs(new_train_labels_path, exist_ok=True)

    # Get list of all image files in the folder
    image_files = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]
    shuffled_image_files = image_files[:]
    random.shuffle(shuffled_image_files)
    zipped_files = list(zip(image_files, shuffled_image_files))

    for tuple in zipped_files:
        mixup2images(tuple[0], tuple[1], alpha)
