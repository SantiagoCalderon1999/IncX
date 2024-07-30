import pickle
from filelock import FileLock
import os
from collections import defaultdict
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

def main():
    load_dotenv()

    azure_storage_connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    azure_container_name = os.environ.get("AZURE_STORAGE_CONTAINER_NAME")
    blob_service_client = BlobServiceClient.from_connection_string(azure_storage_connection_string)
    container_client = blob_service_client.get_container_client(container=azure_container_name)

    data_file = "d_rise.pkl"
    data_file_lock = f'{data_file}.lock'

    blob_names_file = "blob_names.pkl"
    blob_names_file_lock = f'{blob_names_file}.lock'

    with open(blob_names_file, 'rb') as f:
        blob_names_list = pickle.load(f)

    while blob_names_list:
        lock_blob_names = FileLock(blob_names_file_lock, timeout=100)
        with lock_blob_names:
            with open(blob_names_file, 'rb') as f:
                blob_names_list = pickle.load(f)
                if not blob_names_list:
                    break
                current_blob_name = blob_names_list.pop()
                with open(blob_names_file, 'wb') as f:
                    pickle.dump(blob_names_list, f)

        blob_client = container_client.get_blob_client(current_blob_name)
        blob_data = blob_client.download_blob().readall()
        deserialized_data = pickle.loads(blob_data)
        
        lock_data_file = FileLock(data_file_lock, timeout=100)
        with lock_data_file:
            if os.path.exists(data_file):
                with open(data_file, 'rb') as f:
                    metrics_data = pickle.load(f)
            else:
                metrics_data = defaultdict(lambda: defaultdict(list))
            
            model_name = current_blob_name.split('/')[1]
            for entry in deserialized_data:
                metrics_data["insertion"][model_name].append(abs(entry["metrics"]["insertion"]))
                metrics_data["deletion"][model_name].append(abs(entry["metrics"]["deletion"]))
                metrics_data["explanation_time"][model_name].append(abs(entry["metrics"]["explanation_time"]))
                metrics_data["exp_proportion"][model_name].append(abs(entry["metrics"]["exp_proportion"]))
                metrics_data["epg"][model_name].append(abs(entry["metrics"]["epg"]))

            with open(data_file, 'wb') as f:
                pickle.dump(dict(metrics_data), f)

if __name__ == "__main__":
    main()