import numpy as np
import gzip
import urllib.request
import pickle
import os
import matplotlib.pyplot as plt

class MnistData():

    image_size = 28*28
    dataset_dir = 'dataset'
    dataset_pkl = 'mnist.pkl'
    url_base = 'http://jrkwon.com/data/ece5831/mnist/'

    key_file = {
    'train_images': 'train-images-idx3-ubyte.gz',
    'train_labels': 'train-labels-idx1-ubyte.gz',
    'test_images': 't10k-images-idx3-ubyte.gz',
    'test_labels': 't10k-labels-idx1-ubyte.gz'
    }

    def __init__(self):
        self.dataset = {}
        self.dataset_pkl_path = f'{self.dataset_dir}/{self.dataset_pkl}'

        self._init_dataset()

    def _change_one_hot_label(self, y, num_class):
        t = np.zeros((y.size, num_class))
        for idx, row in enumerate(t):
            row[y[idx]] = 1
        
        return t

    def _download(self, file_name):
        file_path = self.dataset_dir + '/' + file_name

        if (os.path.exists(file_path)):
            print(f'File: {file_name} already exists.')
            return
        
        print(f'Downloading {file_name}...')
        urllib.request.urlretrieve(self.url_base + file_name, file_path)
        print('done')
    
    def _download_all(self):
        for file_name in self.key_file.values():
            self._download(file_name)

    def _load_images(self, file_name):
        with gzip.open(file_name, 'rb') as f:
            images = np.frombuffer(f.read(), np.uint8, offset=16)
        images = images.reshape(-1, self.image_size)

        return images
    
    def _load_labels(self, file_name):
        with gzip.open(file_name, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)

        return labels
    
    def _create_dataset(self):
       
        self.dataset['train_images'] = self._load_images(f"{dataset_dir}/{key_file['train_images']}")
        self.dataset['train_labels'] = self._load_labels(f"{dataset_dir}/{key_file['train_labels']}")
        self.dataset['test_images'] = self._load_images(f"{dataset_dir}/{key_file['test_images']}")
        self.dataset['test_labels'] = self._load_labels(f"{dataset_dir}/{key_file['test_labels']}")
        
        with open(f'{self.dataset_pkl_path}', 'wb') as f:
            print(f'Pickle: {self.dataset_dir}/{self.dataset_pkl} is being created')
            pickle.dump(self.dataset, f)
            print('Done')

    def _init_dataset(self):
        self._download_all()
        if os.path.exists(f'{self.dataset_dir}/{self.dataset_pkl}'):
            with open(f'{self.dataset_dir}/{self.dataset_pkl}', 'rb') as f:
                print(f'Pickle: {self.dataset_dir}/{self.dataset_pkl} already exists.')
                print('loading....')
                self.dataset = pickle.load(f)
                print('Done')

        else:
            dataset = _create_dataset()

    def load(self):
        # normalize image datasets
        for key in ('train_images','test_images'):
            self.dataset[key] = self.dataset[key].astype(np.float32)
            self.dataset[key] /= 255.0
        
        # One-hot encoding
        for key in ('train_labels', 'test_labels'):
            self.dataset[key] = self._change_one_hot_label(self.dataset[key], 10)

        return (self.dataset['test_images'], self.dataset['train_labels']), \
            (self.dataset['test_images'], self.dataset['test_labels'])
