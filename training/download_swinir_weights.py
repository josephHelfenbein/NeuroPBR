import os
import urllib.request

# Relative path inside the git submodule
TARGET_DIR = "models/swinir/model_zoo"

# Ensure the folder exists as part of the git submodule setup
if not os.path.exists(TARGET_DIR):
    raise FileNotFoundError(f"{TARGET_DIR} does not exist. Please initialize the submodules.")


files = [
    "001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth",
    "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth",
]

BASE_URL = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"

for fname in files:
    path = os.path.join(TARGET_DIR, fname)
    if not os.path.exists(path):
        print(f"Downloading {fname} ...")
        urllib.request.urlretrieve(BASE_URL + fname, path)
    else:
        print(f"{fname} already exists, skipping.")

print("All files are ready in", TARGET_DIR)
