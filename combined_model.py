import torch
import clean_tabular_data
import clean_images
import os
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, BertConfig, BertModel
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
from datetime import datetime

class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, max_length: int = 50):
        super().__init__()
        self.transform = transform
        self.data = pd.read_csv("Products.csv", lineterminator="\n")
        self.data = clean_tabular_data.clean(self.data)
        self.data["category"] = pd.Categorical(self.data["category"])
        self.class_labels = self.data['category'].cat.categories
        self.data["category"] = self.data["category"].cat.codes
        self.encoder = {name: index for index, name in enumerate(self.class_labels)}
        self.decoder = {index: name for index, name in enumerate(self.class_labels)}
        config = BertConfig()
        config.output_hidden_states = True
        self.model = BertModel(config)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.max_length = max_length
        self.keys = pd.read_csv("Images.csv", lineterminator="\n")
        clean_images.clean_image_data("images/", self.keys)
        self.image_paths = os.listdir('cleaned_images')


    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(f'cleaned_images/{image_path}')
        image_id = image_path[:-4]
        product_id = self.keys[self.keys['id']==image_id].reset_index().loc[0].at['product_id']
        sentence = self.data[self.data['id']==product_id].reset_index().loc[0].at['product_description']
        if self.transform:
            image = self.transform(image)
        example = self.data.iloc[index]
        label = torch.as_tensor(self.encoder[example[5]])
        encoded = self.tokenizer.batch_encode_plus([sentence], max_length=self.max_length, padding='max_length', truncation=True)
        encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
        with torch.no_grad():
            description = self.model(**encoded).last_hidden_state.swapaxes(1,2)
        description = description.squeeze(0)
        return (image, description), label

    def __len__(self):
        return len(self.keys)

class CombinedClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.text_layers = torch.nn.Sequential(
            torch.nn.Conv1d(768, 256, kernel_size=2, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
            torch.nn.Conv1d(256, 128, kernel_size=2, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
            torch.nn.Conv1d(128, 64, kernel_size=2, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2),
            torch.nn.Conv1d(64, 32, kernel_size=2, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(384, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 13)
        )
        self.image_layers = torch.nn.Sequential(
                torch.nn.Conv2d(3, 7, 4),
                torch.nn.ReLU(),
                torch.nn.Conv2d(7, 11, 4),
                torch.nn.ReLU(),
                torch.nn.Flatten(),
                torch.nn.Linear(2816396, 13),
                torch.nn.Softmax(dim=0)
        )
        self.linear = torch.nn.Linear(26, 13)
    
    def forward(self, features):
        image_embedding, text_embedding = features
        image_out = self.image_layers(image_embedding)
        text_out = self.text_layers(text_embedding)
        combined_out = torch.cat((image_out, text_out))
        out = self.linear(combined_out)
        return out

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
    for epoch in range(epochs):
        hist_acc = []
        acc = 0
        for batch in trainloader:
            features, labels = batch
            prediction = model(features)
            loss = F.cross_entropy(prediction, labels)
            loss.backward()
            print(loss.item())
            optimiser.step()
            optimiser.zero_grad()
            acc = torch.sum(torch.argmax(prediction, dim=1) == labels).item()/len(labels)
            hist_acc.append(acc)
            writer.add_scalar('loss', loss.item(), batch_idx)
            batch_idx += 1
        torch.save(model.state_dict(), f'model_evaluation/{timestamp}/weights/epoch{epoch}.pt')

def accuracy(model: CombinedClassifier):
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

if __name__ == "__main__":
    dataset = CombinedDataset()
    batch_size = 8
    test_split = .2
    shuffle_dataset = True
    random_seed= 42
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    trainloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    testloader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
    model = CombinedClassifier()
    train(model)
    acc = accuracy(model)
    if acc >= 0.6:
        torch.save(model.state_dict(), f'final_models/combined_model.pt')