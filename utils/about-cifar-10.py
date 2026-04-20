import pickle
import numpy as np
import matplotlib.pyplot as plt



file_path = "../data/raw/cifar-10-batches-py/data_batch_1"

with open(file_path, "rb") as f:
    batch = pickle.load(f, encoding="bytes")


print(batch.keys())


images = batch[b'data']       # shape: (10000, 3072)
labels = batch[b'labels']

print(images[0][:1500])  # Shows first 10 pixel values only

img = images[0].reshape(3, 32, 32)
img = np.transpose(img, (1, 2, 0))  # CHW → HWC



plt.imshow(img)
plt.title(f"Label: {labels[0]}")
plt.axis("off")
plt.show()



import random

plt.figure(figsize=(10, 6))

for i in range(12):
    idx = random.randint(0, len(images) - 1)

    img = images[idx].reshape(3, 32, 32)
    img = np.transpose(img, (1, 2, 0))

    plt.subplot(3, 4, i + 1)
    plt.imshow(img)
    plt.title(labels[idx])
    plt.axis("off")

plt.tight_layout()
plt.show()




with open("../data/raw/cifar-10-batches-py/batches.meta", "rb") as f:
    meta = pickle.load(f, encoding="bytes")

print(meta)