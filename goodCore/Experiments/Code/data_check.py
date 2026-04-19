#to check validity of the splits used 
import hashlib
import numpy as np


data = np.load("/kaggle/input/datasets/majdoubnourelhouda/medmnist-chest-test/dataset_splits.npz")

X_train_s, Y_train_s = data["X_train"], data["Y_train"]
X_val_s, Y_val_s = data["X_val"], data["Y_val"]
X_test_s, Y_test_s = data["X_test"], data["Y_test"]
def split_checksum(X, Y):
    h = hashlib.md5()
    h.update(X.tobytes())
    h.update(Y.tobytes())
    return h.hexdigest()

print("Train checksum:", split_checksum(X_train_s, Y_train_s))
print("Val   checksum:", split_checksum(X_val_s,   Y_val_s))
print("Test  checksum:", split_checksum(X_test_s,  Y_test_s))

EXPECTED = {
    "train": "3589decea2d85737867535a8a2b72cce",  
    "val":   "7cccf0f16d4c23a7c5f48bee2f2067c1",
    "test":  "ad59c834875716e0921c045341c202c8",
}
assert split_checksum(X_train_s, Y_train_s) == EXPECTED["train"], "Train split changed!"
assert split_checksum(X_val_s,   Y_val_s)   == EXPECTED["val"],   "Val split changed!"
assert split_checksum(X_test_s,  Y_test_s)  == EXPECTED["test"],  "Test split changed!"
print("✓ Splits are identical to the frozen reference")

all_saved   = to_hashes(X_train_s) | to_hashes(X_val_s) | to_hashes(X_test_s)
all_original = to_hashes(X)

assert all_saved == all_original, "FAIL: saved splits don't reconstruct original dataset!"
print("✓ Full coverage — all original samples accounted for")



import pandas as pd

splits = {"Train": Y_train_s, "Val": Y_val_s, "Test": Y_test_s}
label_stats = {}

for name, Y in splits.items():
    label_stats[name] = Y.mean(axis=0)  # prevalence per class

df = pd.DataFrame(label_stats)
df.index = [f"Class {i}" for i in range(Y_train_s.shape[1])]
print(df.round(3))

# Flag any class that's completely missing in val or test
for name, Y in splits.items():
    missing = np.where(Y.sum(axis=0) == 0)[0]
    if len(missing):
        print(f"⚠ {name} is missing classes: {missing}")
    else:
        print(f"✓ {name}: all classes represented")


import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train_s[i].transpose(1, 2, 0)) 
    ax.set_title(str(Y_train_s[i].astype(int)), fontsize=7)
    ax.axis("off")
plt.suptitle("Train samples — verify labels look plausible")
plt.tight_layout()
plt.show()
