import argparse
from incremental_explainer.utils.video import create_video, save_video
import matplotlib.pyplot as plt
from incremental_explainer.tracking_draft import track_saliency_maps
from incremental_explainer.utils.save_results import save_results
from ultralytics import YOLO
import cv2
import pandas as pd
import pickle

def track_saliency_maps_with_video(frame_number, car_set_object, box_index):
    print(f"Frame number: {frame_number}, Car number: {car_set_object}, Explanation index: {box_index}")
    frames, auc_results, _ = track_saliency_maps(frame_number=frame_number, car_number=car_set_object, box_index_first_frame=box_index)
    print(f"Number of frames: {len(frames)}")
    save_results(
        car_number=car_set_object,
        start_frame=frame_number,
        frames_number=len(frames),
        object_index=box_index,
        insertion_move=auc_results
    )

def main(car_number, frame_number):
    for j in range(100):
        model = YOLO("yolov8n.pt")
        with open('results.pkl', 'rb') as f:
            data = pickle.load(f)
        df = pd.DataFrame(data)
        new_frame_number = frame_number + j
        while not df.loc[(df['car_number'] == car_number) & (df['start_frame'] == new_frame_number)].empty:
            frame_number += 1
            new_frame_number = frame_number + j
        image_location = f"datasets/car/car-{car_number}/img/{str(new_frame_number).zfill(8)}.jpg"
        print(f"Processing image: {image_location}")
        img = cv2.imread(image_location)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img, verbose=False)
        number_objects = len(results[0].boxes.cls)
        print(f"Number of objects: {number_objects}")
        for i in range(number_objects):
            track_saliency_maps_with_video(new_frame_number, car_number, i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('car_number', type=int, help='The number of the car to process')
    parser.add_argument('frame_number', type=int, help='The frame number to process')

    args = parser.parse_args()

    main(args.car_number, args.frame_number)
