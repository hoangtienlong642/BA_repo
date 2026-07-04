import urllib.request
import zipfile
import os

url = "https://www.kaggle.com/api/v1/datasets/download/rupakroy/online-payments-fraud-detection-dataset"
zip_path = "online-payments-fraud-detection-dataset.zip"
target_csv = "Synthetic_Financial_datasets_log.csv"

def download_and_extract():
    if not os.path.exists(target_csv):
        print("Downloading dataset (~178MB)... Please wait.")
        try:
            urllib.request.urlretrieve(url, zip_path)
            print("Downloaded successfully!")
        except Exception as e:
            print(f"Error during download: {e}")
            return False

        print("Extracting...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            print("Extracted successfully!")
        except Exception as e:
            print(f"Error during extraction: {e}")
            return False

        # Rename extracted CSV to Synthetic_Financial_datasets_log.csv
        renamed = False
        for file in os.listdir("."):
            if file.endswith(".csv") and file != target_csv and file != "metadata_features.csv":
                os.rename(file, target_csv)
                print(f"Renamed {file} to {target_csv}")
                renamed = True
                break
        
        if not renamed and not os.path.exists(target_csv):
            print("No suitable CSV file found after extraction.")
            return False

        # Remove zip file
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print("Cleaned up zip file.")
    else:
        print(f"File {target_csv} already exists in project root.")

    # Read and print the first 5 lines of the CSV file
    print("\n--- First 5 lines of the dataset ---")
    try:
        with open(target_csv, 'r', encoding='utf-8') as f:
            for _ in range(6): # header + 5 rows
                line = f.readline()
                if not line:
                    break
                print(line.strip())
    except Exception as e:
        print(f"Error reading file preview: {e}")
    return True

if __name__ == "__main__":
    download_and_extract()
