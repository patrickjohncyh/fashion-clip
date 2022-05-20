from torch.utils.data import Dataset
from PIL import Image

class CaptioningDataset(Dataset):
    def __init__(self, captions):
        self.caption = captions

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        caption = self.caption[idx]
        return caption


class ImageDataset(Dataset):

    def __init__(self, images, preprocessing):
        self.images = images
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.preprocessing(self.images[idx])  # preprocess from clip.load

    @classmethod
    def from_path(cls, image_paths, preprocessing):
        images = [Image.open(im_path) for im_path in image_paths]
        return cls(images, preprocessing)
