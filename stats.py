import pandas as pd
import matplotlib.pyplot as plt

# Load result.csv
df = pd.read_csv(
    "runs/detect/train/results.csv",
    skipinitialspace=True,
)

print(df.head())

# Plot
# Header: epoch,         train/box_loss,         train/cls_loss,         train/dfl_loss,   metrics/precision(B),      metrics/recall(B),       metrics/mAP50(B),    metrics/mAP50-95(B),           val/box_loss,           val/cls_loss,           val/dfl_loss,                 lr/pg0,                 lr/pg1,                 lr/pg2

plt.figure(figsize=(20, 10))

plt.subplot(2, 2, 1)
plt.plot(df["epoch"], df["train/box_loss"], label="train/box_loss")
plt.plot(df["epoch"], df["val/box_loss"], label="val/box_loss")
plt.legend()
plt.grid()
plt.xlabel("epoch")
plt.ylabel("box_loss")

plt.subplot(2, 2, 2)
plt.plot(df["epoch"], df["train/cls_loss"], label="train/cls_loss")
plt.plot(df["epoch"], df["val/cls_loss"], label="val/cls_loss")
plt.legend()
plt.grid()
plt.xlabel("epoch")
plt.ylabel("cls_loss")

plt.subplot(2, 2, 3)
plt.plot(df["epoch"], df["train/dfl_loss"], label="train/dfl_loss")
plt.plot(df["epoch"], df["val/dfl_loss"], label="val/dfl_loss")
plt.legend()
plt.grid()
plt.xlabel("epoch")
plt.ylabel("dfl_loss")

plt.subplot(2, 2, 4)
plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="metrics/mAP50(B)")
plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="metrics/mAP50-95(B)")
plt.legend()
plt.grid()
plt.xlabel("epoch")
plt.ylabel("mAP")

plt.show()
