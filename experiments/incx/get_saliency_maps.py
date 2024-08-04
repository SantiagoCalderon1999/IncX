from filelock import FileLock
import cv2
import numpy as np
import time
from PIL import Image
from incrementalexplainer.tracking.increx import IncRex
from incrementalexplainer.models.model_enum import ModelEnum
from incrementalexplainer.models.model_factory import ModelFactory
from incrementalexplainer.explainers.d_rise import DRise
from incrementalexplainer.metrics.saliency_maps.deletion import compute_deletion
from incrementalexplainer.metrics.saliency_maps.insertion import compute_insertion
from incrementalexplainer.metrics.saliency_maps.epg import compute_energy_based_pointing_game
from incrementalexplainer.metrics.saliency_maps.exp_proportion import compute_explanation_proportion
from torchvision import transforms
import time
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import os
import pickle
import joblib
import random

def main():
    jobs_file_lock = "jobs.lock"

    lock_data_file = FileLock(jobs_file_lock, timeout=100)
    jobs = joblib.load("jobs.pkl")
    while len(jobs) > 0:
        
        jobs = joblib.load("jobs.pkl")
        random.shuffle(jobs)
        with lock_data_file:
            job = jobs.pop(0)
            joblib.dump(jobs, "jobs.pkl")

        def resize_image(image_path):
            pil_image = Image.open(image_path)
            image_array = np.array(pil_image)
            return image_array

        load_dotenv()
        AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        AZURE_CONTAINER_NAME = os.environ.get("AZURE_STORAGE_INCREX_CONTAINER_NAME")
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        explainer_name = job[1]
        model_name = job[0]
        k = job[2]

        valid = False
        initial_im = 1
        print(model_name.name)
        model = ModelFactory().get_model(model_name)
        explainer = DRise(model, 1000)
        incRex = IncRex(model, explainer, object_indices=[0])
        while (not valid):
            image_locations = [f'datasets/LASOT/{k}/{str(i).zfill(8)}.jpg' for i in range(initial_im, 301)]

            images = [resize_image(image_location) for image_location in image_locations]
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            img_t = transform(images[0])
            results = model.predict([img_t])
            if len(results[0].class_scores) > 0:
                valid = True
                class_index = np.argmax(results[0].class_scores[0])
                score = results[0].class_scores[0][class_index]
            else:
                initial_im += 1
        print(f"Processing video {k}, score: {score:.2f}")
        
        for i, image in enumerate(images):
            image_location = image_locations[i]
            time_start = time.time()
            results, _ = incRex.explain_frame(image)
            explanation_time = time.time() - time_start
            result = results[0]
            if result.current_index == -1:
                continue
            
            results_dict = {
                "metrics": {
                    "explanation_time": explanation_time
                },
                "detection": {
                    "class_index": int(result.current_index)
                },
                "maps": {
                    "saliency_map": result.saliency_map,
                    "mask": result.mask
                }
            }
            
            results_array_bytes = pickle.dumps(results_dict)
            blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=f"{explainer_name.name}/{model_name.name}/{image_location.split('/')[-3]}/{image_location.split('/')[-2]}/{image_location.split('/')[-1].split('.')[0]}.pkl")
            blob_client.upload_blob(results_array_bytes, overwrite = True)
        jobs = joblib.load("jobs.pkl")
            
if __name__ == "__main__":
    main()
