import kagglehub
import os

os.environ["KAGGLEHUB_CACHE"] = "./dataset"
path = kagglehub.dataset_download("nih-chest-xrays/sample")

print("Path to dataset files:", path)