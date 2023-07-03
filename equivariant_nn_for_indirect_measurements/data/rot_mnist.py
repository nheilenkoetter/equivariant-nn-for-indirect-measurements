import os
import torchvision.transforms as transforms
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import numpy as np
from torchvision.datasets.utils import download_and_extract_archive

class MnistRotDataset(VisionDataset):
    """ Official RotMNIST dataset, Code from LieConv.
        https://github.com/mfinzi/LieConv """

    ignored_index = -100
    class_weights = None
    balanced = True
    stratify = True
    num_targets=10
    resources = ["http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip"]
    training_file = 'mnist_all_rotation_normalized_float_train_valid.amat'
    test_file = 'mnist_all_rotation_normalized_float_test.amat'

    def __init__(self, root, train=True, transform=None, download=True):
        #normalize = transforms.Normalize((mean,), (std,))
        #transform = transforms.Compose([transform])

        super().__init__(root,transform=transform)
        self.train = train
        if download:
            self.download()
        if train:
            file=os.path.join(self.raw_folder, self.training_file)
        else:
            file=os.path.join(self.raw_folder, self.test_file)
        
        self.transform = transform

        data = np.loadtxt(file, delimiter=' ')
            
        self.images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32)
        self.labels = data[:, -1].astype(np.int64)
        self.num_samples = len(self.labels)
    
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.raw_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.raw_folder,
                                            self.test_file)))
    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')
    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder,exist_ok=True)
        os.makedirs(self.processed_folder,exist_ok=True)

        # download files
        for url in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=None)
        print('Downloaded!')

    def __len__(self):
        return len(self.labels)

#    def default_aug_layers(self):
#        return RandomRotateTranslate(0)# no translation