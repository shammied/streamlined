import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tqdm import tqdm
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device available for running: ")
print(device)

class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNN, self).__init__()
    
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        
        out = self.fc1(x)
        out = self.relu1(out)

        out = self.fc2(out)
        out = self.relu2(out)

        out = self.fc3(out)
        return out


if __name__ == "__main__":
    
    train_data = pd.read_pickle("train_df.p")
    dev_data = pd.read_pickle("dev_df.p")
    test_data = pd.read_pickle("test_df.p")

    train_labels = train_data["label"]
    le = LabelEncoder()
    new_labels = le.fit_transform(train_labels)

    train_data["new_label"] = new_labels
    dev_data["new_label"] = le.transform(dev_data["label"])
    test_data["new_label"] = le.transform(test_data["label"])

    train_data = train_data.drop('label', axis=1).sample(frac=1).reset_index(drop=True)
    dev_data = dev_data.drop('label', axis=1).sample(frac=1).reset_index(drop=True)
    test_data = test_data.drop('label', axis=1).sample(frac=1).reset_index(drop=True)


    input_dim = 1*20
    hidden_dim = 100
    output_dim = 8
    batch_size = 1024
    learning_rate = 0.01
    num_epochs = 400

    starts = list(range(0, len(train_data), batch_size))

    model = FeedforwardNN(input_dim, hidden_dim, output_dim)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    iter = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for start in tqdm(starts):
            end = min(len(train_data), start + batch_size)
            batch = train_data[start:end]
            
            optimizer.zero_grad()
    
            outputs = model(torch.tensor(batch.drop('new_label', axis=1).values.astype(np.float32)).to(device))
    
            labels = torch.LongTensor(batch["new_label"].values.astype(np.float32)).to(device)
        
            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            epoch_loss += loss

            iter += 1

        print(f"Epoch {epoch} avg train loss: {epoch_loss/len(starts)}")
                


    test_preds = []

    with torch.no_grad():
        for index, row in test_data.iterrows():
        probs = model(torch.tensor(row[:-1].values.astype(np.float32)).to(device))
        test_preds.append(torch.argmax(probs))

    tp = [t.cpu() for t in test_preds]
    print(classification_report(test_data["new_label"],tp))
        