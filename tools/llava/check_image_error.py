import os
from PIL import Image
import threading
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
Image.MAX_IMAGE_PIXELS = 5000000000
from io import BytesIO
# from petrel_client.client import Client
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES=True 

def check_image_size(image_path, lock, error_file):
    try:
        image_path = image_path.replace("langchao:s3://multi_modal/playground/data/", "/cpfs01/shared/gmai/medical_preprocessed/image-text/general_data/").replace("image-text/report_generation/", "report_generation/").replace("s3://", "/cpfs01/shared/gmai/").replace("/mnt/petrelfs/share_data/wangweiyun/share_data_eval/", "/cpfs01/shared/gmai/medical_preprocessed/image-text/general_data/")
        with Image.open(image_path) as img:
            size = img.size
            if size[0] < 3:
                with lock:
                    with open(error_file, 'a+') as f:
                        f.write(f"{image_path}\n")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        with lock:
            with open(error_file, 'a+') as f:
                f.write(f"{image_path}\n")

def process_images(image_paths, max_threads=60):
    lock = threading.Lock()
    error_file = '/cpfs01/shared/gmai/xtuner_lite_workspace/error_images.txt'

    with ThreadPoolExecutor(max_threads) as executor:
        with tqdm(total=len(image_paths)) as pbar:
            futures = [executor.submit(check_image_size, path, lock, error_file) for path in image_paths]
            
            for future in as_completed(futures):
                future.result()  # This will raise any exceptions caught during processing
                pbar.update(1)


with open("/cpfs01/shared/gmai/xtuner_lite_workspace/xtuner/scripts/pretrain_data_0822.json") as f:
    json_files = json.load(f)
for ith, (name, item) in enumerate(json_files.items()):
    # 示例图像路径列表
    image_paths = []
    file_path = item["annotations"]
    print(ith, file_path)
    for img in json.load(open(file_path)):
        if "image" in img:
            image_paths.append(img["image"])
    process_images(image_paths)