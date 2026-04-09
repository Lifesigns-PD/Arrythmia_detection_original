import glob
import json
from collections import Counter

files = glob.glob('testing_final/mV_extracts_output/*.json')
res = Counter()
conf_sum = 0
conf_count = 0

for f in files:
    try:
        data = json.load(open(f))
        for pkg in data:
            for s in pkg['analysis']['segments']:
                lbl = s.get('ectopy_label', 'Unknown')
                conf = s.get('ectopy_confidence', 0.0)
                res[lbl] += 1
                if lbl != "None":
                    conf_sum += conf
                    conf_count += 1
    except: pass

print(f"Ectopy Label Counts: {dict(res)}")
if conf_count > 0:
    print(f"Average Confidence for Non-None: {conf_sum / conf_count:.4f}")
else:
    print("No Non-None labels found.")
