import json
from datetime import datetime
import numpy as np

data = json.load(open('ECG_Data_Extracts/ADM640316196.json'))
print(f"Total packets: {len(data)}")

timestamps = []
for p in data[:20]:
    ts_str = p['utcTimestamp']['$date'].replace("Z", "+00:00")
    timestamps.append(datetime.fromisoformat(ts_str))
    
deltas = []
for i in range(1, len(timestamps)):
    deltas.append((timestamps[i] - timestamps[i-1]).total_seconds())

print(f"Deltas: {deltas[:5]}")
print(f"Median delta: {np.median(deltas)}")
samples_per_packet = len(data[0]['value'][0])
print(f"Samples per packet: {samples_per_packet}")
print(f"Estimated fs: {samples_per_packet / np.median(deltas)}")
