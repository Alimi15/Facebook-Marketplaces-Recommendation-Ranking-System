import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import clean_tabular_data
import clean_images
import os
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import ToTensor
from datetime import datetime

class TextDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.data = pd.read_csv("Products.csv", lineterminator="\n")
        self.data = clean_tabular_data.clean(self.data)
        self.data, self.encoder = clean_tabular_data.encode(self.data)

    def __getitem__(self, index):
        example = self.data.iloc[index]
        features = example[[2, 4, -1]]
        label = example[3]
        return (features, label)

    def __len__(self):
        return len(self.data)

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        super().__init__()
        self.transform = transform
        self.data = pd.read_csv("Products.csv", lineterminator="\n")
        self.data = clean_tabular_data.clean(self.data)
        self.data, self.encoder = clean_tabular_data.encode(self.data)
        self.data["category"] = pd.Categorical(self.data["category"])
        self.class_labels = self.data['category'].cat.categories
        self.keys = pd.read_csv("Images.csv", lineterminator="\n")
        clean_images.clean_image_data("images/", self.keys)
        self.image_paths = os.listdir('cleaned_images')


    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(f'cleaned_images/{image_path}')
        image_id = image_path[:-4]
        product_id = self.keys[self.keys['id']==image_id].reset_index().loc[0].at['product_id']
        class_index = self.data[self.data['id']==product_id].reset_index().loc[0].at['category']
        if self.transform:
            image = self.transform(image)
        return image, class_index

    def __len__(self):
        return len(self.keys)

class ResNetNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
        self.layers.fc = torch.nn.Linear(2048, 13)
    
    def forward(self, features):
        return F.softmax(self.layers(features), dim=0)

    def predict_probs(self, features):
        with torch.no_grad():
            return self.forward(features)

    def predict(self, features):
        probs = self.predict_probs(features)
        idx = np.argmax(probs)
        return idx

class CNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
                torch.nn.Conv2d(3, 7, 4),
                torch.nn.ReLU(),
                torch.nn.Conv2d(7, 11, 4),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(2816396, 13),
                torch.nn.Softmax(dim=0)
        )
    
    def forward(self, features):
        return self.layers(features)

    def predict_probs(self, features):
        with torch.no_grad():
            return self.forward(features)

    def predict(self, features):
        probs = self.predict_probs(features)
        idx = np.argmax(probs)
        return idx
    
def train(model: torch.nn.Module, epochs=10):
    optimiser = torch.optim.SGD(model.parameters(), lr=0.001)
    writer = SummaryWriter()
    batch_idx = 0
    timestamp = str(datetime.now()).replace(' ', '-').replace(':', '-').replace('.', '-')
    os.mkdir(f'model_evaluation/{timestamp}')
    os.mkdir(f'model_evaluation/{timestamp}/weights')
    for i, epoch in enumerate(range(epochs)):
        for batch in trainloader:
            features, labels = batch
            prediction = model(features)
            loss = F.cross_entropy(prediction, labels)
            loss.backward()
            print(loss.item())
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar('loss', loss.item(), batch_idx)
            batch_idx += 1
        torch.save(model.state_dict(), f'model_evaluation/{timestamp}/weights/epoch{i}.pt')

def accuracy(model: CNN, test_set):
    test_features = torch.Tensor()
    test_labels = torch.Tensor()
    for batch in testloader:
        features, labels = batch
        test_features = torch.cat((test_features, features))
        test_labels = torch.cat((test_labels, labels))
    y_pred = model.predict(test_features)
    y_true = test_labels
    acc = np.array(y_pred==y_true).sum() / len(y_true)
    return acc

if __name__ == '__main__':
    image_dataset = ImageDataset(transform=ToTensor())
    batch_size = 8
    test_split = .2
    shuffle_dataset = True
    random_seed= 42
    dataset_size = len(image_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    trainloader = DataLoader(image_dataset, batch_size=batch_size, sampler=train_sampler)
    testloader = DataLoader(image_dataset, batch_size=batch_size, sampler=test_sampler)
    model = CNN()
    train(model)
    acc = accuracy(model)
    if acc >= 0.6:
        torch.save(model.state_dict(), f'final_models/image_model.pt')