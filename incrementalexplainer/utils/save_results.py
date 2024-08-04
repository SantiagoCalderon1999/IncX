import pickle
import os

PICKLE_FILE = "results.pkl"


def save_results(
    car_number: int,
    start_frame: int,
    frames_number: int,
    object_index: int,
    insertion_move: list[int],
):
    row = {
        "car_number": car_number,
        "start_frame": start_frame,
        "frames_number": frames_number,
        "insertion_move": insertion_move,
        "object_index": object_index,
    }

    if os.path.exists(PICKLE_FILE):
        with open(PICKLE_FILE, "rb") as file:
            data = pickle.load(file)
            data.append(row)
        with open(PICKLE_FILE, "wb") as file:
            pickle.dump(data, file)
    else:
        # Create the pickle file and write default data to it
        with open(PICKLE_FILE, "wb") as file:
            pickle.dump([row], file)
            data = PICKLE_FILE
