from utils.dataset import MaskDataset
import os
import glob
from PIL import Image

DATASET_PATH = "/Users/yifan/Downloads/face_mask_detection"
CLASSES = {"with_mask": 0,
           "without_mask": 1,
           "mask_weared_incorrect": 2}


def prepare_data(dataset_path, output_path):
    images_path = os.path.join(dataset_path, "images")
    xml_path = os.path.join(dataset_path, "annotations")

    for dir_name in list(CLASSES.keys()):
        path = os.path.join(output_path, dir_name)
        if not os.path.exists(path):
            os.makedirs(path)

    for image_name in os.listdir(images_path):
        bboxes, labels = MaskDataset.read_xml(image_name, xml_path)
        for idx, data in enumerate(zip(bboxes, labels)):
            bbox, label = data
            label_str = list(CLASSES.keys())[label]
            img = Image.open(os.path.join(images_path, image_name))
            cropped = img.crop(bbox)
            cropped = cropped.resize((128, 128))
            # cropped.show()
            fname, ext = image_name.split(".")
            save_path = os.path.join(output_path, label_str, f"{fname}_{idx}.{ext}")
            cropped.save(save_path)


if __name__ == "__main__":
    prepare_data(DATASET_PATH, "./mask_dataset_processed")

