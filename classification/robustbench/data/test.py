import os
import numpy as np

root = '/home/xuxun/suyongyi/Dataset/ImageNet-C/brightness/1'

# all_files = []

paths = os.listdir(root)

f = open("imagenet_test_image_ids_100000.txt", "wb+")

for path in paths:
    abs_path = os.path.join(root, path)
    if os.path.isdir(abs_path):
        all_files = os.listdir(abs_path)
        np.random.shuffle(all_files)
        for file in all_files[:10]:
            if file.upper().endswith(".JPEG"):
                # all_files.append(os.path.join(path, file))
                f.write(os.path.join(path, file).encode("UTF-8"))
                f.write(b"\n")
    else:
        continue


f.close()