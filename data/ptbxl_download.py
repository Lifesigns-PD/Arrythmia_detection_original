import os
import requests
import zipfile

# Path setup
download_dir = 'raw_data/ptbxl'
os.makedirs(download_dir, exist_ok=True)
zip_path = os.path.join(download_dir, 'ptbxl.zip')

# Official PhysioNet Direct Link for the ZIP (More reliable than the wfdb downloader)
url = "https://physionet.org/static/publishedprojects/ptb-xl/1.0.3/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"

print("Downloading PTB-XL via direct link (approx 3.0 GB)...")
print("This is more reliable than the Python library for this specific dataset.")

try:
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    print("✅ Download complete. Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(download_dir)
    
    # Cleanup the zip file to save space
    os.remove(zip_path)
    print(f"✅ PTB-XL is ready in: {download_dir}")

except Exception as e:
    print(f"❌ Download failed: {e}")
    print("If this fails, please download the ZIP manually from: https://physionet.org/content/ptb-xl/1.0.3/")