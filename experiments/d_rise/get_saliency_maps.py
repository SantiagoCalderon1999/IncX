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
from dotenv import load_dotenv

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
    D_RISE_RESULTS_FOLDER_PATH = os.environ.get("D_RISE_RESULTS_FOLDER_PATH")
    
    incx_results_list = find_files(INCX_RESULTS_FOLDER_PATH)
    d_rise_results_list = find_files(D_RISE_RESULTS_FOLDER_PATH)
    
    incx_not_d_rise = set(incx_results_list) - set(d_rise_results_list)
    incx_not_d_rise = list(incx_not_d_rise)

    random.shuffle(incx_not_d_rise)

    while len(incx_not_d_rise) > 0:
        file_location = incx_not_d_rise.pop(0)
        
        with open(file_location, "rb") as f:
            incx_file = pickle.load(f)
        current_index = incx_file["detection"]["current_index"]
        
        file_location = file_location.replace("\\", "/")
        model_name = file_location.split("/")[-4]
        video_index = file_location.split("/")[-2]
        image_index = int(file_location.split("/")[-1].split(".")[0])
        
        print(f"Processing {file_location}")
        print(f"Current index: {current_index}, model: {model_name}, video index: {video_index}, image index: {image_index}")
        
        image_path = f"../../datasets/LASOT/{video_index}/{str(image_index).zfill(8)}.jpg"

        model = ModelFactory().get_model(ModelEnum[model_name])
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([transforms.ToTensor()])
        img_t = transform(img)

        start_time = time.time()
        results = model.predict([img_t])
        explainer = ExplainerFactory(model).get_explainer(ExplainerEnum.D_RISE)
        start_time = time.time()
        saliency_maps = explainer.create_saliency_map(
            np.array(Image.open(image_path))
        )
        class_index = np.argmax(results[0].class_scores[current_index].detach())
        bounding_box = np.array(
            results[0].bounding_boxes[current_index].cpu().detach()
        )
        saliency_map = saliency_maps[current_index]
        _, _, mask = compute_initial_sufficient_explanation(
            model, saliency_map, img, class_index, bounding_box, divisions=100
        )
        
        explanation_time = time.time() - start_time
        
        results_dict = {
            "metrics": {"explanation_time": explanation_time},
            "detection": {
                "bounding_box": bounding_box,
                "class_index": int(class_index),
                "class_score": max(
                    results[0].class_scores[current_index].detach().numpy()
                ),
            },
            "maps": {"saliency_map": saliency_map, "mask": mask},
        }
        file_name = f"{image_path.split('/')[-1].split('.')[0]}.pkl"
            
        file_path = f"{D_RISE_RESULTS_FOLDER_PATH}/D_RISE/{model_name}/{image_path.split('/')[-3]}/{image_path.split('/')[-2]}/"
        full_path = os.path.join(file_path, file_name)

        os.makedirs(file_path, exist_ok=True)
        
        with open(full_path, "wb") as f:
            pickle.dump(results_dict, f)
            
        print(f"Saved on {full_path}")
        
        incx_results_list = find_files(INCX_RESULTS_FOLDER_PATH)
        d_rise_results_list = find_files(D_RISE_RESULTS_FOLDER_PATH)
        
        incx_not_d_rise = set(incx_results_list) - set(d_rise_results_list)
        incx_not_d_rise = list(incx_not_d_rise)
        random.shuffle(incx_not_d_rise)

if __name__ == "__main__":
    main()