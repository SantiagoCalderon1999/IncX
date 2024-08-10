import os
import random
import joblib
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from incx.models.model_enum import ModelEnum


def find_files(directory="."):
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    return file_list

def main():
    load_dotenv()
    INCX_RESULTS_FOLDER_PATH = os.environ.get("INCX_RESULTS_FOLDER_PATH")
    
    blob_names = find_files(INCX_RESULTS_FOLDER_PATH)
    
    blob_names = [blob_name.replace("\\", "/") for blob_name in blob_names]
    blob_names = [
        '/'.join(name.split('/')[-5:]) 
        for name in blob_names
    ]
    random.shuffle(blob_names)

    joblib.dump(blob_names, "blob_names_metrics.pkl")

    import numpy as np

    results = {
        "D-RISE": {
            ModelEnum.YOLO.name: {
                "Insertion": np.zeros((10, 300)),
                "Deletion": np.zeros((10, 300)),
                "EPG": np.zeros((10, 300)),
                "Explanation Proportion": np.zeros((10, 300)),
                "Time": np.zeros((10, 300)),
            },
            ModelEnum.RT_DETR.name: {
                "Insertion": np.zeros((10, 300)),
                "Deletion": np.zeros((10, 300)),
                "EPG": np.zeros((10, 300)),
                "Explanation Proportion": np.zeros((10, 300)),
                "Time": np.zeros((10, 300)),
            },
            ModelEnum.FASTER_RCNN.name: {
                "Insertion": np.zeros((10, 300)),
                "Deletion": np.zeros((10, 300)),
                "EPG": np.zeros((10, 300)),
                "Explanation Proportion": np.zeros((10, 300)),
                "Time": np.zeros((10, 300)),
            },
        },
        "Incx": {
            ModelEnum.YOLO.name: {
                "Insertion": np.zeros((10, 300)),
                "Deletion": np.zeros((10, 300)),
                "EPG": np.zeros((10, 300)),
                "Explanation Proportion": np.zeros((10, 300)),
                "Time": np.zeros((10, 300)),
            },
            ModelEnum.RT_DETR.name: {
                "Insertion": np.zeros((10, 300)),
                "Deletion": np.zeros((10, 300)),
                "EPG": np.zeros((10, 300)),
                "Explanation Proportion": np.zeros((10, 300)),
                "Time": np.zeros((10, 300)),
            },
            ModelEnum.FASTER_RCNN.name: {
                "Insertion": np.zeros((10, 300)),
                "Deletion": np.zeros((10, 300)),
                "EPG": np.zeros((10, 300)),
                "Explanation Proportion": np.zeros((10, 300)),
                "Time": np.zeros((10, 300)),
            },
        },
    }
    joblib.dump(results, "metrics_results.pkl")

    print(
        "Blob names have been saved to 'blob_names_metrics.pkl and comparison results array has been initialized 'metrics_results.pkl'."
    )


if __name__ == "__main__":
    main()
