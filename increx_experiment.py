from datasets.dataset_enum import DatasetEnum
import os
from incremental_explainer.tracking.increx  import IncRex
from incremental_explainer.models.model_enum import ModelEnum
from incremental_explainer.models.model_factory import ModelFactory
import numpy as np
from PIL import Image
from incremental_explainer.models.labels import coco_labels
from incremental_explainer.metrics.saliency_maps.deletion import compute_deletion
from incremental_explainer.metrics.saliency_maps.insertion import compute_insertion
from incremental_explainer.metrics.saliency_maps.epg import compute_energy_based_pointing_game
from incremental_explainer.metrics.saliency_maps.exp_proportion import compute_explanation_proportion
from incremental_explainer.models.model_enum import ModelEnum
from incremental_explainer.explainers.explainer_enum import ExplainerEnum
from incremental_explainer.explainers.explainer_factory import ExplainerFactory
import time
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import pickle

if __name__ == "__main__":
    load_dotenv()
    AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    AZURE_CONTAINER_NAME = os.environ.get("AZURE_STORAGE_INCREX_CONTAINER_NAME")
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    print(AZURE_CONTAINER_NAME)
    container_client = blob_service_client.get_container_client(container=AZURE_CONTAINER_NAME)
        
    for explainer_name in ExplainerEnum:
        for model_name in ModelEnum:
            model = ModelFactory().get_model(model_name)
            explainer = ExplainerFactory(model).get_explainer(explainer_name)
            for dataset in DatasetEnum:
                if dataset == DatasetEnum.LASOT:
                    dataset_path = os.path.join(".", "datasets", dataset.name)
                    for i, path in enumerate(os.listdir(dataset_path)):
                        images = os.listdir(os.path.join(dataset_path, path))
                        image_set_name = os.path.join(dataset_path, path).replace("\\", "_").replace(".", "")
                        image_locations = [os.path.join(dataset_path, path, image).replace('\\', '/') for image in images]
                        image_locations
                        incRex = IncRex(model, explainer)
                        
                        divisions = 100
                        
                        for image_location in image_locations:
                            
                            image = np.array(Image.open(image_location))
                            start_time = time.time()
                            results, _ = incRex.explain_frame(image)
                            explanation_time = time.time() - start_time
                            results_array = []
                            for _, result in results.items():
                                class_index = coco_labels.index(result.label)
                                deletion = compute_deletion(model, result.saliency_map, image, class_index, result.bounding_box, divisions = divisions)
                                insertion = compute_insertion(model, result.saliency_map, image, class_index, result.bounding_box, divisions = divisions)
                                epg = compute_energy_based_pointing_game(result.saliency_map, result.bounding_box)
                                
                                exp_prop = compute_explanation_proportion(result.mask)
                                results_dict = {
                                    "metrics": {
                                        "deletion": deletion,
                                        "insertion": insertion,
                                        "epg": epg,
                                        "exp_proportion": exp_prop,
                                        "explanation_time": explanation_time
                                    },
                                    "detection": {
                                        "bounding_box": result.bounding_box,
                                        "class_index": int(class_index),
                                        "class_score": result.score
                                    },
                                    "maps": {
                                        "saliency_map": result.saliency_map,
                                        "sufficient_explanation": result.sufficient_explanation,
                                        "mask": result.mask
                                    }
                                }
                                results_array.append(results_dict)
                            results_array_bytes = pickle.dumps(results_array)
                            blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=f"{explainer_name.name}/{model_name.name}/{image_location.split('/')[-3]}/{image_location.split('/')[-2]}/{image_location.split('/')[-1].split('.')[0]}.pkl")
                            blob_client.upload_blob(results_array_bytes)
                            print(results_array)