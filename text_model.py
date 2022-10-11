import torch
import clean_tabular_data
import os
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, BertConfig, BertModel
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from datetime import datetime

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, max_length: int = 50):
        super().__init__()
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


    def __getitem__(self, index):
        example = self.data.iloc[index]
        label = torch.as_tensor(self.encoder[example[5]])
        sentence = example[4]
        encoded = self.tokenizer.batch_encode_plus([sentence], max_length=self.max_length, padding='max_length', truncation=True)
        encoded = {key:torch.LongTensor(value) for key, value in encoded.items()}
        with torch.no_grad():
            description = self.model(**encoded).last_hidden_state.swapaxes(1,2)
        description = description.squeeze(0)
        return description, label

    def __len__(self):
        return len(self.data)

class TextClassifier(torch.nn.Module):
    def __init__(self, input_size: int = 768):
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Conv1d(input_size, 256, kernel_size=2, stride=1, padding=1),
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
    
    def forward(self, features):
        return self.main(features)

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
    os.mkdir(f'text_model_evaluation/{timestamp}')
    os.mkdir(f'text_model_evaluation/{timestamp}/weights')
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
            if acc > 0.6:
                torch.save(model.state_dict(), f'text_model_evaluation/{timestamp}/weights/epoch{epoch}.pt')

def accuracy(model: TextClassifier):
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
    text_dataset = TextDataset()
    batch_size = 8
    test_split = .2
    shuffle_dataset = True
    random_seed= 42
    dataset_size = len(text_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    trainloader = DataLoader(text_dataset, batch_size=batch_size, sampler=train_sampler)
    testloader = DataLoader(text_dataset, batch_size=batch_size, sampler=test_sampler)
    model = TextClassifier()
    train(model)
    acc = accuracy(model)
    if acc >= 0.6:
        torch.save(model.state_dict(), f'final_models/text_model.pt')