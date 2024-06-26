from datasets.dataset_enum import DatasetEnum
import os
import random
from incremental_explainer.models.model_enum import ModelEnum
from incremental_explainer.explainers.explainer_enum import ExplainerEnum
import pickle
import cv2
from incremental_explainer.models.model_factory import ModelFactory

from incremental_explainer.explainers.explainer_factory import ExplainerFactory
import torchvision.transforms as transforms
import numpy as np
from incremental_explainer.metrics.deletion import compute_deletion
from incremental_explainer.metrics.insertion import compute_insertion
from incremental_explainer.metrics.epg import compute_energy_based_pointing_game
from incremental_explainer.metrics.exp_proportion import compute_explanation_proportion
import time

if __name__ == "__main__":

    images_paths = []

    for dataset in DatasetEnum:
        partial_images_paths = []
        dataset_path = os.path.join(".", "datasets", dataset.name)
        for path in os.listdir(dataset_path):
            images = os.listdir(os.path.join(dataset_path, path))
            partial_images_paths += [os.path.join(dataset_path, path, image).replace('\\', '/') for image in images]
        print(f"{dataset.name}: {len(partial_images_paths)} images")
        images_paths.extend(partial_images_paths)
    
    print('---------------------------------')
    print(f"Total {len(images_paths)} images")
    
    random.shuffle(images_paths)

    job_array = []

    for explainer in ExplainerEnum:
        for model in ModelEnum:
            for image_path in images_paths:
                job_array.append((explainer.name, model.name, image_path))

    random.shuffle(job_array)

    pickle_file_name = './results/baseline.pkl'

    curr_pickle = {}

    if os.path.exists(pickle_file_name):
        with open(pickle_file_name, 'rb') as file:
            curr_pickle = pickle.load(file)
    job_pickle = set(job_array) - set(curr_pickle.keys())
    
    for job_key in job_pickle:
        explainer_name, model_name, image_location = job_key


        if os.path.exists(pickle_file_name):
            with open(pickle_file_name, 'rb') as file:
                curr_pickle = pickle.load(file)

        if job_key in set(curr_pickle.keys()):
            print(f"Skipping image: {image_location}, model: {model_name}, explainer: {explainer_name}")
            continue

        print(f"Processing image: {image_location}, model: {model_name}, explainer: {explainer_name}")
        model = ModelFactory().get_model(ModelEnum[model_name])
        img = cv2.imread(image_location)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img_t = transform(img)

        start_time = time.time()
        results = model.predict([img_t])
        explainer = ExplainerFactory(results).get_explainer(ExplainerEnum[explainer_name])
        saliency = explainer.create_saliency_map(image_location, model)
        explanation_time = time.time() - start_time

        results_array = []

        divisions = 100
        objects_number = len(results[0].class_scores)
        for object_index in range(objects_number):
            print(f"Started metrics: {image_location} for index {object_index} / {len(results[0].class_scores) - 1}")
            class_index = np.argmax(results[0].class_scores[object_index].detach())
            saliency_map = np.array(saliency[object_index]['detection']).transpose(1, 2, 0)[:,:,0]
            bounding_box = np.array(results[0].bounding_boxes[object_index].cpu().detach())
            deletion = compute_deletion(model, saliency_map, img, class_index, bounding_box, divisions = divisions)
            insertion = compute_insertion(model, saliency_map, img, class_index, bounding_box, divisions = divisions)
            epg = compute_energy_based_pointing_game(saliency_map, bounding_box)
            exp_prop, suf_exp = compute_explanation_proportion(model, saliency_map, img, class_index, bounding_box, divisions = divisions)
            print(f"Finished metrics: {image_location} for index {object_index} / {len(results[0].class_scores) - 1}")

            results_dict = {
                "metrics": {
                    "deletion": deletion,
                    "insertion": insertion,
                    "epg": epg,
                    "exp_proportion": exp_prop,
                    "explanation_time": explanation_time
                },
                "detection": {
                    "bounding_box": bounding_box,
                    "class_index": int(class_index),
                    "class_score": max(results[0].class_scores[object_index].detach().numpy())
                }
            }
            results_array.append(results_dict)

        curr_pickle = {}

        if os.path.exists(pickle_file_name):
            with open(pickle_file_name, 'rb') as file:
                curr_pickle = pickle.load(file)

        curr_pickle[job_key] = results_array

        with open(pickle_file_name, 'wb') as file:
            pickle.dump(curr_pickle, file)
            
        print(f"Finished image: {image_location}, model: {model_name}, explainer: {explainer_name}")
        print('---------------------------------')