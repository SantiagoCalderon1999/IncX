import pickle
from incx.models.model_enum import ModelEnum
from incx.metrics.saliency_maps.deletion import compute_deletion
from incx.metrics.saliency_maps.insertion import compute_insertion
from incx.metrics.saliency_maps.epg import (
    compute_energy_based_pointing_game,
)
from incx.metrics.saliency_maps.exp_proportion import (
    compute_explanation_proportion,
)
from collections import defaultdict
from IPython.display import clear_output
import random
import joblib
from filelock import FileLock
import numpy as np
from incx.models.model_factory import ModelFactory
from PIL import Image

from azure.storage.blob import BlobServiceClient
import os
from dotenv import load_dotenv

load_dotenv()

AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = os.environ.get("AZURE_STORAGE_INCREX_CONTAINER_NAME")
blob_service_client = BlobServiceClient.from_connection_string(
    AZURE_STORAGE_CONNECTION_STRING
)

container_client_incx = blob_service_client.get_container_client(
    container=AZURE_CONTAINER_NAME
)
blob_names_incx = [blob.name for blob in container_client_incx.list_blobs()]
print(len(blob_names_incx))

AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = os.environ.get("AZURE_STORAGE_CONTAINER_NAME")
blob_service_client = BlobServiceClient.from_connection_string(
    AZURE_STORAGE_CONNECTION_STRING
)

container_client_drise = blob_service_client.get_container_client(
    container=AZURE_CONTAINER_NAME
)
blob_names_drise = [blob.name for blob in container_client_drise.list_blobs()]
print(len(blob_names_drise))

total_pearson_coeff = defaultdict(float)
total_structural_similarity_index = defaultdict(float)
total_scanpath_saliency = defaultdict(float)
total_dice_coefficient = defaultdict(float)
total_jaccard_index = defaultdict(float)


blob_name_file_lock = "blob_names_metrics.lock"
lock_blobs_name_comparison = FileLock(blob_name_file_lock, timeout=100)

random.shuffle(blob_names_incx)
num_blobs = 0

comparison_file_lock = "metrics_comparison.lock"
lock_comparison = FileLock(comparison_file_lock, timeout=100)

results = []

with lock_blobs_name_comparison:
    blob_names_incx = joblib.load("blob_names_metrics.pkl")

def read_image(image_path):
    pil_image = Image.open(image_path)
    image_array = np.array(pil_image)
    return image_array


while blob_names_incx:
    with lock_blobs_name_comparison:
        blob_name = blob_names_incx.pop()
        joblib.dump(blob_names_incx, "blob_names_metrics.pkl")

    # Download and load blob data for INCX
    blob_client = container_client_incx.get_blob_client(blob_name)
    blob_bytes = blob_client.download_blob().readall()
    dict_incx = pickle.loads(blob_bytes)
    model_name = blob_name.split("/")[1]
    index = dict_incx["detection"]["current_index"]

    # Download and load blob data for DRISE
    blob_client = container_client_drise.get_blob_client(blob_name)
    blob_bytes = blob_client.download_blob().readall()
    array_drise = pickle.loads(blob_bytes)
    dict_drise = array_drise[index]

    model_name = blob_name.split("/")[1]
    current_index = dict_incx["detection"]["current_index"]
    class_index = dict_drise["detection"]["class_index"]

    blob_client = container_client_drise.get_blob_client(blob_name)
    blob_bytes = blob_client.download_blob().readall()
    array_drise = pickle.loads(blob_bytes)
    dict_drise = array_drise[current_index]

    saliency_map_incx = dict_incx["maps"]["saliency_map"]
    saliency_map_drise = dict_drise["maps"]["saliency_map"]

    model_enum_el = ModelEnum[model_name]
    model = ModelFactory().get_model(model_enum_el)

    frame_number = int(blob_name.split("/")[-1].split(".")[0])
    video_number = blob_name.split("/")[-2]
    dataset = "LASOT"

    image_path = f"../../datasets/LASOT/{video_number}/{str(frame_number).zfill(8)}.jpg"
    image = read_image(image_path)

    bounding_box = dict_drise["detection"]["bounding_box"]
    if class_index > 80:
        continue

    deletion_drise = compute_deletion(
        model=model,
        saliency_map=saliency_map_drise,
        image=image,
        class_index=class_index,
        bounding_box=bounding_box,
        object_index=current_index,
        divisions=100,
    )
    deletion_incx = compute_deletion(
        model=model,
        saliency_map=saliency_map_incx,
        image=image,
        class_index=class_index,
        bounding_box=bounding_box,
        object_index=current_index,
        divisions=100,
    )

    insertion_drise = compute_insertion(
        model=model,
        saliency_map=saliency_map_drise,
        image=image,
        class_index=class_index,
        bounding_box=bounding_box,
        object_index=current_index,
        divisions=100,
    )

    insertion_incx = compute_insertion(
        model=model,
        saliency_map=saliency_map_incx,
        image=image,
        class_index=class_index,
        bounding_box=bounding_box,
        object_index=current_index,
        divisions=100,
    )

    epg_drise = compute_energy_based_pointing_game(
        saliency_map=saliency_map_drise, bounding_box=bounding_box
    )

    epg_incx = compute_energy_based_pointing_game(
        saliency_map=saliency_map_incx, bounding_box=bounding_box
    )

    time_drise = dict_drise["metrics"]["explanation_time"]
    time_incx = dict_incx["metrics"]["explanation_time"]
    mask_incx = dict_incx["maps"]["mask"]
    mask_drise = dict_drise["maps"]["mask"]

    exp_proportion_drise = compute_explanation_proportion(mask_drise)
    exp_proportion_incx = compute_explanation_proportion(mask_incx)

    frame_number = int(blob_name.split("/")[-1].split(".")[0]) - 1
    image_number = int(blob_name.split("/")[-2]) - 1
    with lock_comparison:
        metrics_results = joblib.load("metrics_results.pkl")
        metrics_results["D-RISE"][model_name]["Insertion"][image_number][
            frame_number
        ] = insertion_drise
        metrics_results["D-RISE"][model_name]["Deletion"][image_number][
            frame_number
        ] = deletion_drise
        metrics_results["D-RISE"][model_name]["EPG"][image_number][frame_number] = (
            epg_drise
        )
        metrics_results["D-RISE"][model_name]["Explanation Proportion"][image_number][
            frame_number
        ] = exp_proportion_drise
        metrics_results["D-RISE"][model_name]["Time"][image_number][frame_number] = (
            time_drise
        )

        metrics_results["Incx"][model_name]["Insertion"][image_number][frame_number] = (
            insertion_incx
        )
        metrics_results["Incx"][model_name]["Deletion"][image_number][frame_number] = (
            deletion_incx
        )
        metrics_results["Incx"][model_name]["EPG"][image_number][frame_number] = (
            epg_incx
        )
        metrics_results["Incx"][model_name]["Explanation Proportion"][image_number][
            frame_number
        ] = exp_proportion_incx
        metrics_results["Incx"][model_name]["Time"][image_number][frame_number] = (
            time_incx
        )
        joblib.dump(metrics_results, "metrics_results.pkl")

    clear_output(wait=True)
    print(f"Deletion DRiSE: {deletion_drise}")
    print(f"Deletion IncX: {deletion_incx}")
    print(f"Insertion DRiSE: {insertion_drise}")
    print(f"Insertion IncX: {insertion_incx}")
    print(f"EPG DRiSE: {epg_drise}")
    print(f"EPG IncX: {epg_incx}")
    print(f"Exp Proportion DRiSE: {exp_proportion_drise}")
    with lock_blobs_name_comparison:
        blob_names_incx = joblib.load("blob_names_metrics.pkl")
    random.shuffle(blob_names_incx)
