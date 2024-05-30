import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import shutil

BASE_DIR = "DATASETS_LOW_RES"

# Labels are in train/labels/(file_name).txt in the format: class x_center y_center width height
labels_dir = os.path.join(BASE_DIR, "train/labels")
labels = os.listdir(labels_dir)
# print(len(labels))

# Load the labels into a pandas dataframe
df = pd.DataFrame(columns=["file_name", "class", "x_center",
              "y_center", "width", "height"])

def load_labels(labels_dir):
  df = pd.DataFrame(columns=["file_name", "class", "x_center",
              "y_center", "width", "height"])
  for label in labels:
    with open(os.path.join(labels_dir, label), "r") as f:
      lines = f.readlines()
      for line in lines:
        line = line.strip().split(" ")
        if len(line) != 5:
          continue
        
        new_row = {"file_name": label.split(".")[0], 
              "class": line[0],
              "x_center": line[1], 
              "y_center": line[2], 
              "width": line[3], 
              "height": line[4]}
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
  return df

df = load_labels(labels_dir)

# Load example image and display
img_name = "POSAD_1_11-Sep_09-43-28"
img_2_name = "Monitoring_photo_04-Mar_04-31-35"
# Filter the dataframe to only include the bounding boxes for the image we loaded
img_boxes = df[df["file_name"] == img_name]
img_2_boxes = df[df["file_name"] == img_2_name]

img = plt.imread(os.path.join(BASE_DIR, "train/images/", img_name + ".jpg"))
img_2 = plt.imread(os.path.join(BASE_DIR, "train/images/", img_2_name + ".jpg"))

# plt.imshow(img)
# plt.axis("off")

# print(img.shape)
# print(img_boxes)

# Display bounding boxes
def disp_bb(img, img_boxes):
  for index, row in img_boxes.iterrows():
    # Values are normalized
    x_center = float(row["x_center"]) * img.shape[1]
    y_center = float(row["y_center"]) * img.shape[0]
    width = float(row["width"]) * img.shape[1]
    height = float(row["height"]) * img.shape[0]
    
    x1 = x_center - width/2
    x2 = x_center + width/2
    y1 = y_center - height/2
    y2 = y_center + height/2
    
    plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color="red")
  plt.imshow(img)
  plt.axis("off")  
  plt.show()  

# disp_bb(img, img_boxes)
# plt.show()

def get_random_image_patch(img, img_boxes, width, height):
    # display the image
    try:
        random_box = img_boxes.sample()    
        x_center = float(random_box.iloc[0]["x_center"]) * img.shape[1]
        y_center = float(random_box.iloc[0]["y_center"]) * img.shape[0]
    except ValueError:
        # create a random box
        x_center = random.uniform(0, 1)
        y_center = random.uniform(0, 1)
        
    x1 = x_center - width / 2
    x2 = x_center + width / 2
    y1 = y_center - height / 2
    y2 = y_center + height / 2
    
    # to integers
    x1, x2, y1, y2 = map(int, [x1, x2, y1, y2])
    
    cropped_img = img[y1:y2, x1:x2]

    return cropped_img

def cutmix(img, img_boxes, img2, img_2_boxes, threshold=0.3):
    img = np.copy(img)

    for i, bb in img_boxes.iterrows():
        # original bounding box
        x_center = float(bb["x_center"]) * img.shape[1]
        y_center = float(bb["y_center"]) * img.shape[0]
        width_bb = float(bb["width"])  * img.shape[1]
        height_bb = float(bb["height"])  * img.shape[0]
        
        # new bounding box - 30 percent of the original bounding box
        reduction_factor = np.sqrt(threshold)
        width = width_bb * reduction_factor
        height = height_bb * reduction_factor
        
        x1 = x_center - width / 2
        x2 = x_center + width / 2
        y1 = y_center - height / 2
        y2 = y_center + height / 2
        
        # move the new bounding box to a random location within the original bounding box
        random_x = random.uniform(0, width_bb/1.9)
        random_y = random.uniform(0, height_bb/1.9)
        
        img_x1 = x1 + random_x
        img_x2 = x2 + random_x
        img_y1 = y1 + random_y
        img_y2 = y2 + random_y
        
        # map to integers
        img_x1, img_x2, img_y1, img_y2 = map(int, [img_x1, img_x2, img_y1, img_y2])
                
        # loop through rectangle coords and set them to 0
        for y in range(img_y1, img_y2):
            for x in range(img_x1, img_x2):
                if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                    img[y, x] = [0, 0, 0]
                    
        img_2 = get_random_image_patch(img2, img_2_boxes, img_x2 - img_x1, img_y2 - img_y1)

        # paste the new image onto the original image
        for y in range(img_y1, img_y2):
            for x in range(img_x1, img_x2):
                if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                    if 0<= y - img_y1 < img_2.shape[0] and 0<= x - img_x1 < img_2.shape[1]:
                        img[y, x] = img_2[y - img_y1, x - img_x1]
       
        # plot the new bounding box - patched
        #plt.plot([img_x1, img_x2, img_x2, img_x1, img_x1], [img_y1, img_y1, img_y2, img_y2, img_y1], color="blue")
        
                    
    # y labels stay the same
    return img, img_boxes

def mixup(img, img_boxes, img2, img_2_boxes, lam=0.5):
    img = np.copy(img)
    img = img.astype(np.float32) / 255.0
    img2 = np.copy(img2)
    img2 = img2.astype(np.float32) / 255.0
    # new image
    img = lam*img + (1 - lam)*img2
    # concatenate labels from the 2 labels
    img_boxes = pd.concat([img_boxes, img_2_boxes])
    # print(img.shape)
    # print(img_boxes)
    return img, img_boxes

# img, img_boxes = mixup(img, img_boxes, img_2, img_2_boxes)
# disp_bb(img=img, img_boxes=img_boxes)

def cutout(img, img_boxes, threshold=0.3):
    img = np.copy(img)

    for i, bb in img_boxes.iterrows():
        # original bounding box
        x_center = float(bb["x_center"]) * img.shape[1]
        y_center = float(bb["y_center"]) * img.shape[0]
        width_bb = float(bb["width"])  * img.shape[1]
        height_bb = float(bb["height"])  * img.shape[0]
        
        # new bounding box - 30 percent of the original bounding box
        width = width_bb * threshold
        height = height_bb * threshold
        
        x1 = x_center - width / 2
        x2 = x_center + width / 2
        y1 = y_center - height / 2
        y2 = y_center + height / 2
        
        # move the new bounding box to a random location within the original bounding box
        random_x = random.uniform(0, width_bb/1.5)
        random_y = random.uniform(0, height_bb/1.5)
        
        x1 = x1 + random_x
        x2 = x2 + random_x
        y1 = y1 + random_y
        y2 = y2 + random_y
        
        # loop through rectangle coords
        for y in range(int(y1), int(y2)):
            for x in range(int(x1), int(x2)):
                if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                    img[y, x] = [0, 0, 0]
                    
    # y labels stay the same
    return img, img_boxes

# AUG_DIR = "DATASETS_COMBINE_ALLFOUR"

# Loop through dataset and either don't apply any augmentation, apply cutmix, cutout, mixup or a combination of all three
# For both test, train
def generate_data_set(options = ["cutmix", "cutout", "mixup", "none"], AUG_DIR = "DATASETS_COMBINE_ALLFOUR"):
   for split in ["train"]: # leave test set and val set unchanged / no augmentation applied to them
    df_labels = df
    images_dir = os.path.join(BASE_DIR, split, "images")
    images_aug_dir = os.path.join(AUG_DIR, split, "images")
    images = os.listdir(images_dir) #list all filenames inside images_dir
    labels_aug_dir = os.path.join(AUG_DIR, split, "labels")

    os.makedirs(images_aug_dir, exist_ok=True)
    os.makedirs(labels_aug_dir, exist_ok=True)

    for image in tqdm(images, desc=f"Processing {split} set"): # for each image in images_dir = <base>/<split>/images
        
        img = plt.imread(os.path.join(images_dir, image)) # read current image as np array
        img_boxes = df_labels[df_labels["file_name"] == image.split(".")[0]] # get all bounding boxes of the current image
        
        # randomly pick img_2 
        img_2_name = random.choice(images)
        img_2 = plt.imread(os.path.join(images_dir, img_2_name))
        img_2_boxes = df_labels[df_labels["file_name"] == img_2_name.split(".")[0]]

        aug = random.choice(options)
        if aug == "cutmix":
           img, boxes = cutmix(img, img_boxes, img_2, img_2_boxes)
        #    disp_bb(img, boxes)
           plt.imsave(os.path.join(images_aug_dir, image), img)
            # Save bounding boxes to txt
            # Drop the first column
           boxes = boxes.drop(columns=["file_name"])
           boxes.to_csv(os.path.join(AUG_DIR, split, "labels", image.split(".")[0] + ".txt"), header=False, index=False, sep=" ")
        #    break
        elif aug == "cutout":
           img, boxes = cutout(img, img_boxes)
        #    disp_bb(img, boxes)
           plt.imsave(os.path.join(images_aug_dir, image), img)
           boxes = boxes.drop(columns=["file_name"])
           boxes.to_csv(os.path.join(AUG_DIR, split, "labels", image.split(".")[0] + ".txt"), header=False, index=False, sep=" ")
        #    break
        elif aug == "mixup":
           img, boxes = mixup(img, img_boxes, img_2, img_2_boxes)
        #    disp_bb(img, boxes)
           plt.imsave(os.path.join(images_aug_dir, image), img)
           boxes = boxes.drop(columns=["file_name"])
           boxes.to_csv(os.path.join(AUG_DIR, split, "labels", image.split(".")[0] + ".txt"), header=False, index=False, sep=" ")
        #    break
        elif aug == "none":
        #    disp_bb(img, img_boxes)
           plt.imsave(os.path.join(images_aug_dir, image), img)
           img_boxes = img_boxes.drop(columns=["file_name"])
           img_boxes.to_csv(os.path.join(AUG_DIR, split, "labels", image.split(".")[0] + ".txt"), header=False, index=False, sep=" ")
        #    break
        else:
           raise ValueError("No valid augmentation chosen! make sure `options` has the right augmentation options!")

    # Copy over warpd.yaml
    shutil.copy(os.path.join(BASE_DIR, "warpd.yaml"), os.path.join(AUG_DIR, "warpd.yaml"))
    # Copy over val and test directories
    for folder in ["val", "test"]:
        src_dir = os.path.join(BASE_DIR, folder)
        dest_dir = os.path.join(AUG_DIR, folder)
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        shutil.copytree(src_dir, dest_dir)
    print("Finished processing")


# generates random augmentation dataset
generate_data_set(AUG_DIR = "DATASETS_RANDOM_AUG")

# generates all mixup dataset
# generate_data_set(options=["mixup"], AUG_DIR = "DATASETS_MIXUP")

# generates all cutout dataset
# generate_data_set(options=["cutout"], AUG_DIR = "DATASETS_CUTOUT")

# generates all cutmix dataset
# generate_data_set(options=["cutmix"], AUG_DIR = "DATASETS_CUTMIX")

# generates all no augmentation  dataset
# generate_data_set(options=["none"], AUG_DIR = "DATASETS_NO_AUGMENTATION")
