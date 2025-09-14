'''
This script aims to extract the desired videos s3d features from the pre obtained features by Howto100M team
'''
import json
import os
import subprocess
from tqdm import tqdm


museums = json.load(open('../museums.json', 'r'))

list_videos = list()
for m in museums:
    for r in m['rooms']:
        list_videos.extend(m['rooms'][r])

print('Num total vids', len(list_videos))
list_videos = list(set(list_videos))
print('Num unique vids', len(list_videos))

path_zip_file = '/data01/aabdari/projects/Agr_dataset_Collection/s3d/howto100m_s3d_features.zip'
path_s3d_features = 's3d_features'
os.makedirs(path_s3d_features, exist_ok=True)
list_failed = list()
_list = list()


for v in tqdm(list_videos):
    # command = f'7z e /data01/aabdari/projects/Agr_dataset_Collection/s3d/howto100m_s3d_features.zip howto100m_s3d_features/{v}.mp4.npy -os3d_features'
    command = ["7z", "e", path_zip_file, f"howto100m_s3d_features/{v}.webm.npy", f"-o{path_s3d_features}"]
    if not os.path.isfile(path_s3d_features + os.sep + v + '.mp4.npy') and not os.path.isfile(path_s3d_features + os.sep + v + '.webm.npy'):
        print(v)
        _list.append(v)
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print("Extraction successful:", result.stdout)
        except subprocess.CalledProcessError as e:
            print("Error occurred:", e.stderr)
            list_failed.append(v)
            print('len list failed', len(list_failed))

print(_list)
print(list_failed)
print('len list failed', len(list_failed))
