from datasets.dataset_enum import DatasetEnum
import os
import random
from incx.models.model_enum import ModelEnum
from incx.explainers.explainer_enum import ExplainerEnum
import pickle
import cv2
from incx.models.model_factory import ModelFactory

from incx.explainers.explainer_factory import ExplainerFactory
import torchvision.transforms as transforms
import numpy as np

from incx.utils.explanations import (
    compute_initial_sufficient_explanation,
)
import time
from PIL import Image
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

if __name__ == "__main__":
    images_paths = []

    for dataset in DatasetEnum:
        partial_images_paths = []
        dataset_path = os.path.join(".", "datasets", dataset.name)
        for path in os.listdir(dataset_path):
            images = os.listdir(os.path.join(dataset_path, path))
            partial_images_paths += [
                os.path.join(dataset_path, path, image).replace("\\", "/")
                for image in images
            ]
        print(f"{dataset.name}: {len(partial_images_paths)} images")
        images_paths.extend(partial_images_paths)

    print("---------------------------------")
    print(f"Total {len(images_paths)} images")

    random.shuffle(images_paths)

    job_array = []

    for explainer in ExplainerEnum:
        for model in ModelEnum:
            for image_path in images_paths:
                job_array.append((explainer.name, model.name, image_path))

    random.shuffle(job_array)

    curr_pickle = {}
    load_dotenv()
    AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    AZURE_CONTAINER_NAME = os.environ.get("AZURE_STORAGE_CONTAINER_NAME")
    blob_service_client = BlobServiceClient.from_connection_string(
        AZURE_STORAGE_CONNECTION_STRING
    )

    container_client = blob_service_client.get_container_client(
        container=AZURE_CONTAINER_NAME
    )

    blob_list = container_client.list_blobs()

    finished_jobs = []
    for blob in blob_list:
        array = blob.name.split("/")
        finished_jobs.append(
            (
                array[0],
                array[1],
                "../../datasets/" + "/".join(array[2:]).split(".")[0] + ".jpg",
            )
        )
    print(f"Total jobs: {len(job_array)}")
    print(f"Finished jobs: {len(finished_jobs)}")
    print(f"Finished jobs: {finished_jobs}")
    job_pickle = set(job_array) - set(finished_jobs)
    print(f"Remaining jobs: {len(job_pickle)}")
    for job_key in job_pickle:
        explainer_name, model_name, image_location = job_key

        finished_jobs = []
        blob_list = container_client.list_blobs()
        for blob in blob_list:
            array = blob.name.split("/")
            finished_jobs.append(
                (
                    array[0],
                    array[1],
                    "../../datasets/" + "/".join(array[2:]).split(".")[0] + ".jpg",
                )
            )

        if job_key in set(finished_jobs):
            print(
                f"Skipping image: {image_location}, model: {model_name}, explainer: {explainer_name}"
            )
            continue
        print(f"Finished jobs: {len(finished_jobs)}")
        print(
            f"Processing image: {image_location}, model: {model_name}, explainer: {explainer_name}"
        )
        model = ModelFactory().get_model(ModelEnum[model_name])
        img = cv2.imread(image_location)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([transforms.ToTensor()])
        img_t = transform(img)

        start_time = time.time()
        results = model.predict([img_t])
        explainer = ExplainerFactory(model).get_explainer(ExplainerEnum[explainer_name])
        saliency_maps = explainer.create_saliency_map(
            np.array(Image.open(image_location))
        )

        divisions = 100
        objects_number = len(results[0].class_scores)
        for object_index in range(objects_number):
            saliency_map = saliency_maps[object_index]
            class_index = np.argmax(results[0].class_scores[object_index].detach())
            bounding_box = np.array(
                results[0].bounding_boxes[object_index].cpu().detach()
            )
            suf_expl, _, mask = compute_initial_sufficient_explanation(
                model, saliency_map, img, class_index, bounding_box, divisions=divisions
            )

        explanation_time = time.time() - start_time

        results_array = []

        for object_index in range(objects_number):
            print(
                f"Started metrics: {image_location} for index {object_index} / {len(results[0].class_scores) - 1}"
            )
            class_index = np.argmax(results[0].class_scores[object_index].detach())
            bounding_box = np.array(
                results[0].bounding_boxes[object_index].cpu().detach()
            )
            saliency_map = saliency_maps[object_index]
            print(
                f"Finished metrics: {image_location} for index {object_index} / {len(results[0].class_scores) - 1}"
            )

            results_dict = {
                "metrics": {"explanation_time": explanation_time},
                "detection": {
                    "bounding_box": bounding_box,
                    "class_index": int(class_index),
                    "class_score": max(
                        results[0].class_scores[object_index].detach().numpy()
                    ),
                },
                "maps": {"saliency_map": saliency_map, "mask": mask},
            }
            results_array.append(results_dict)

        results_array_bytes = pickle.dumps(results_array)

        blob_client = blob_service_client.get_blob_client(
            container=AZURE_CONTAINER_NAME,
            blob=f"{explainer_name}/{model_name}/{image_location.split('/')[-3]}/{image_location.split('/')[-2]}/{image_location.split('/')[-1].split('.')[0]}.pkl",
        )
        blob_client.upload_blob(results_array_bytes)

        print(
            f"Finished image: {image_location}, model: {model_name}, explainer: {explainer_name}"
        )
        print("---------------------------------")
