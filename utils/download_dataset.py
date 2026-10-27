import os
import zipfile
import urllib.request
import shutil

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

def setup_tiny_imagenet(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")
    extract_target = os.path.join(data_dir, "tiny-imagenet-200")

    # Download
    if not os.path.exists(extract_target):
        if not os.path.exists(zip_path):
            print(f"Download di Tiny ImageNet da {TINY_IMAGENET_URL}...")
            urllib.request.urlretrieve(TINY_IMAGENET_URL, zip_path)
            print("Download completato.")
        
        print("Estrazione archivio...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Estrazione completata.")
        if os.path.exists(zip_path):
            os.remove(zip_path)

    # Riorganizzazione della cartella val per ImageFolder
    val_dir = os.path.join(extract_target, "val")
    val_images_dir = os.path.join(val_dir, "images")
    val_annotations_file = os.path.join(val_dir, "val_annotations.txt")

    if os.path.exists(val_images_dir):
        print("Riorganizzazione cartella val...")
        val_img_dict = {}
        with open(val_annotations_file, "r") as f:
            for line in f.readlines():
                parts = line.strip().split("\t")
                val_img_dict[parts[0]] = parts[1]

        for img_name, class_id in val_img_dict.items():
            class_dir = os.path.join(val_dir, class_id)
            os.makedirs(class_dir, exist_ok=True)
            src = os.path.join(val_images_dir, img_name)
            dst = os.path.join(class_dir, img_name)
            if os.path.exists(src):
                shutil.move(src, dst)

        shutil.rmtree(val_images_dir)
        print("Cartella val configurata correttamente per ImageFolder.")

if __name__ == "__main__":
    setup_tiny_imagenet()