import os
import random
import joblib
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from incx.models.model_enum import ModelEnum


def main():
    load_dotenv()

    azure_storage_connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    azure_container_name = os.environ.get("AZURE_STORAGE_INCREX_CONTAINER_NAME")

    if not azure_storage_connection_string or not azure_container_name:
        raise ValueError(
            "Environment variables AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_CONTAINER_NAME are not set."
        )

    blob_service_client = BlobServiceClient.from_connection_string(
        azure_storage_connection_string
    )

    container_client = blob_service_client.get_container_client(
        container=azure_container_name
    )

    blob_names = [blob.name for blob in container_client.list_blobs()]
    random.shuffle(blob_names)

    joblib.dump(blob_names, "blob_names.pkl")

    import numpy as np

    results = {
        ModelEnum.YOLO.name: {
            "Pearson": np.zeros((10, 300)),
            "Structural": np.zeros((10, 300)),
            "Scanpath": np.zeros((10, 300)),
            "Dice": np.zeros((10, 300)),
            "Jaccard": np.zeros((10, 300)),
        },
        ModelEnum.RT_DETR.name: {
            "Pearson": np.zeros((10, 300)),
            "Structural": np.zeros((10, 300)),
            "Scanpath": np.zeros((10, 300)),
            "Dice": np.zeros((10, 300)),
            "Jaccard": np.zeros((10, 300)),
        },
        ModelEnum.FASTER_RCNN.name: {
            "Pearson": np.zeros((10, 300)),
            "Structural": np.zeros((10, 300)),
            "Scanpath": np.zeros((10, 300)),
            "Dice": np.zeros((10, 300)),
            "Jaccard": np.zeros((10, 300)),
        },
    }
    joblib.dump(results, "comparison_results.pkl")

    print(
        "Blob names have been saved to 'blob_names.pkl and comparison results array has been initialized 'comparison_results.pkl'."
    )


if __name__ == "__main__":
    main()
