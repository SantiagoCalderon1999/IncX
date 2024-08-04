import os
import random
import joblib
import pickle
from collections import defaultdict
from dotenv import load_dotenv
from tqdm import tqdm
from filelock import FileLock
from azure.storage.blob import BlobServiceClient
from incrementalexplainer.metrics.comparison.explanations.dice_coefficient import compute_dice_coefficient
from incrementalexplainer.metrics.comparison.explanations.jaccard_index import compute_jaccard_index
from incrementalexplainer.metrics.comparison.saliency_maps.pearson_coefficient import compute_pearson_coefficient
from incrementalexplainer.metrics.comparison.saliency_maps.structural_similarity_index import compute_structural_similarity_index
from IPython.display import clear_output

def main():
    # Load environment variables
    load_dotenv()

    # Setup Azure blob clients for INCX and DRISE containers
    azure_storage_connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    container_name_incx = os.environ.get("AZURE_STORAGE_INCREX_CONTAINER_NAME")
    blob_service_client = BlobServiceClient.from_connection_string(azure_storage_connection_string)
    container_client_incx = blob_service_client.get_container_client(container=container_name_incx)

    container_name_drise = os.environ.get("AZURE_STORAGE_CONTAINER_NAME")
    container_client_drise = blob_service_client.get_container_client(container=container_name_drise)

    # Load blob names
    blob_names_incx = joblib.load("blob_names.pkl")
    random.shuffle(blob_names_incx)

    # Setup file locks
    blob_name_file_lock = "blob_names_comparison.lock"
    lock_blobs_name_comparison = FileLock(blob_name_file_lock, timeout=100)

    comparison_file_lock = "comparison.lock"
    lock_comparison = FileLock(comparison_file_lock, timeout=100)
    # Process each blob
    num_blobs = 0
    while blob_names_incx:
        with lock_blobs_name_comparison:
            blob_name = blob_names_incx.pop()
            joblib.dump(blob_names_incx, "blob_names.pkl")
        print(f"Processing blob {blob_name}")

        clear_output(wait=True)

        # Download and load blob data for INCX
        blob_client = container_client_incx.get_blob_client(blob_name)
        blob_bytes = blob_client.download_blob().readall()
        dict_incx = pickle.loads(blob_bytes)
        model_name = blob_name.split("/")[1]
        index = dict_incx["detection"]["class_index"]

        # Download and load blob data for DRISE
        blob_client = container_client_drise.get_blob_client(blob_name)
        blob_bytes = blob_client.download_blob().readall()
        array_drise = pickle.loads(blob_bytes)
        dict_drise = array_drise[index]

        # Extract maps for comparison
        saliency_map_incx = dict_incx["maps"]["saliency_map"]
        saliency_map_drise = dict_drise["maps"]["saliency_map"]

        # Compute metrics
        pearson_coeff = compute_pearson_coefficient(saliency_map_incx, saliency_map_drise)
        structural_similarity_index = compute_structural_similarity_index(saliency_map_incx, saliency_map_drise)

        mask_incx = dict_incx["maps"]["mask"]
        mask_drise = dict_drise["maps"]["mask"]

        dice_coefficient = compute_dice_coefficient(mask_incx, mask_drise)
        jaccard_index = compute_jaccard_index(mask_incx, mask_drise)

        num_blobs += 1

        # Update results
        frame_number = int(blob_name.split("/")[-1].split(".")[0]) - 1
        image_number = int(blob_name.split("/")[-2]) - 1

        with lock_comparison:
            comparison_results = joblib.load("comparison_results.pkl")
            comparison_results[model_name]['Pearson'][image_number][frame_number] = pearson_coeff
            comparison_results[model_name]['Structural'][image_number][frame_number] = structural_similarity_index
            comparison_results[model_name]['Dice'][image_number][frame_number] = dice_coefficient
            comparison_results[model_name]['Jaccard'][image_number][frame_number] = jaccard_index
            joblib.dump(comparison_results, "comparison_results.pkl")

        print(f"Finished blob {blob_name}")
        with lock_blobs_name_comparison:
            blob_names_incx = joblib.load("blob_names.pkl")
        random.shuffle(blob_names_incx)

if __name__ == "__main__":
    main()