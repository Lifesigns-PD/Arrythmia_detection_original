import wfdb
import os

download_dir = 'raw_data/mitdb'
os.makedirs(download_dir, exist_ok=True)

print("Starting download of MIT-BIH Arrhythmia Database...")
# mitdb is much smaller than PTB-XL, this should be fast!
wfdb.dl_database('mitdb', dl_dir=download_dir)

print(f"✅ Download complete! Data saved to: {download_dir}")