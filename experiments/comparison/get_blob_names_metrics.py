import os
import random
import joblib
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from incrementalexplainer.models.model_enum import ModelEnum


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
