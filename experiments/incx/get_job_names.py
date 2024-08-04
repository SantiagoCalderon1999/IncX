from incx.models.model_enum import ModelEnum
from incx.explainers.explainer_enum import ExplainerEnum
import joblib


def main():
    combinations_array = []
    for explainer_name in ExplainerEnum:
        for model_name in ModelEnum:
            for image_index in range(1, 11):
                combinations_array.append((model_name, explainer_name, image_index))

    joblib.dump(combinations_array, "jobs.pkl")


if __name__ == "__main__":
    main()
