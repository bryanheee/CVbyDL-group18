# CVbyDL-group18

- Tested on: Python 3.9.6
- It is recommended to create a virtual environment first before proceeding with the next step
- There is a requirements.txt
```
 pip install -r requirements.txt
```

## Dividing original train set into train subset and val(idation) subset
I splitted the Warp-D training set into a training subset and a subset for validation, by running `train_val_split.py`. 

This is done so that YOLO does not use the test set as a validation set, otherwise it would overfit on the testset. 

After unzipping the WARP dataset archive that you can download from kaggle in the root of your `pwd` (current working directory), you should be able to run `train_val_split.py`, in order to obtain the needed structure for the `datasets` directory, see below.

## Datasets Structure

For this project to run correctly, the `datasets` folder needs to have the following structure:

```plaintext
datasets/
├── test/
│   ├── images/
│   └── labels/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── warpd.yaml
```

## Applying a data augmentation to the WaRP-D dataset

For mixup, run `mixup.py`.
Note mixup is only applied to the train subset, not the val subset or test set
To make yolo use the other dataset, as opposed to the non-augmented dataset, you can run `yolo setting` in your terminal and it should print the location of `settings.yaml` on your machine and the current values of the settings, e.g.:
```
...

Printing '<location of settings.yaml will be displayed here>'

settings_version: 0.0.4
datasets_dir: <absolute path to the  datasets dir (mentioned in previous section) here>
weights_dir: weights
runs_dir: runs
uuid: 9c5094d29dfc2925e1212b859ed9624ad9a42e4b7c67cd515f0f84eb99d5f1f6
sync: true
api_key: ''
openai_api_key: ''
clearml: true
comet: true
dvc: true
hub: true
mlflow: true
neptune: true
raytune: true
tensorboard: true
wandb: true
```

If you want to now make YOLOv8 train on the augmented dataset instead of the original dataset, you must first navigate to `settings.yaml` and open it in an editor and change the current value of `datasets_dir` to the new directory created by `mixup.py` or `<any other data augmentation>.py`

Now having done the above, you must add `data.yaml` (just manually copy and paste) to the location specified in the previous section. Also do the same for the `val` and `test` directories (copy them into the newly created directory)

Then to commence training, run `train.py` in your terminal.