from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torchvision.transforms as transforms
import numpy as np
import os
import xml.etree.ElementTree as ET
from PIL import Image


class MaskDataset(Dataset):
    CLASSES = {"with_mask": 0,
               "without_mask": 1,
               "mask_weared_incorrect": 2}

    def __init__(self, dataset_path, transform=None):
        self.image_dir = os.path.join(dataset_path, "images")
        self.xml_dir = os.path.join(dataset_path, "annotations")
        self.image_list = os.listdir(self.image_dir)
        self.transform = transform

    def __getitem__(self, idx):
        """
        Load an image and an annotation
        """
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        bbox, labels = self.read_xml(img_name, self.xml_dir)
        boxes = torch.as_tensor(bbox, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = dict()

        target['boxes'] = boxes
        target['label'] = labels
        target['image_id'] = torch.tensor([idx])

        return img, target

    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def read_xml(file_name, xml_dir):
        """
        Function used to get the bounding boxes and labels from the xml file
        Input:
            file_name: image file name
            xml_dir: directory of xml file
        Return:
            bbox : list of bounding boxes
            labels: list of labels
        """
        bboxes = []
        labels = []

        annot_path = os.path.join(xml_dir, file_name[:-3] + 'xml')
        tree = ET.parse(annot_path)
        root = tree.getroot()
        for boxes in root.iter('object'):
            ymin = int(boxes.find("bndbox/ymin").text)
            xmin = int(boxes.find("bndbox/xmin").text)
            ymax = int(boxes.find("bndbox/ymax").text)
            xmax = int(boxes.find("bndbox/xmax").text)
            label = boxes.find('name').text

            label_idx = MaskDataset.CLASSES[label]
            bboxes.append([xmin, ymin, xmax, ymax])

            labels.append(label_idx)

        return bboxes, labels


def collate_fn(batch):
    return tuple(zip(*batch))


def get_data_loaders(batch_size, dataset_path, num_workers=1):
    """ Splits the data into training, validation
    and testing datasets. Returns data loaders for the three preprocessed datasets.

    Args:
        batch_size: A int representing the number of samples per batch

    Returns:
        train_loader: iterable training dataset organized according to batch size
        val_loader: iterable validation dataset organized according to batch size
        test_loader: iterable testing dataset organized according to batch size
        classes: A list of strings denoting the name of each class
    """

    classes = []
    ########################################################################
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].
    transform = transforms.Compose(
        [transforms.Resize((128, 128)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # Load gestures training data
    dataset = MaskDataset(dataset_path, transform=transform)
    # Get the list of indices to sample from
    relevant_indices = list(range(len(dataset)))

    # Split into train and validation and test
    np.random.seed(1000)  # Fixed numpy random seed for reproducible shuffling
    np.random.shuffle(relevant_indices)
    split_train = int(len(relevant_indices) * 0.7)
    split_test = int(len(relevant_indices) * 0.9)

    # split into training and validation indices
    relevant_train_indices, relevant_val_indices, relevant_test_indices = \
        relevant_indices[:split_train], relevant_indices[split_train:split_test], relevant_indices[split_test:]
    train_sampler = SubsetRandomSampler(relevant_train_indices)


    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               num_workers=num_workers, sampler=train_sampler,
                                               collate_fn=collate_fn)
    val_sampler = SubsetRandomSampler(relevant_val_indices)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             num_workers=num_workers, sampler=val_sampler,
                                             collate_fn=collate_fn)

    test_sampler = SubsetRandomSampler(relevant_test_indices)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              num_workers=num_workers, sampler=test_sampler,
                                              collate_fn=collate_fn)
    return train_loader, val_loader, test_loader, classes
