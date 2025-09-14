import torch

class MyDataset:
    def __init__(self) -> None:
        self.enviroment = []
        self.labels = []
        self.latest_enviroment = []
        self.latest_labels = []

    def add(self, enviroment, labels):
        self.enviroment.append(enviroment)
        self.labels.append(labels)
        self.latest_enviroment = enviroment
        self.latest_labels = labels

    def get_batch(self, batch_size=1):
        indices = torch.randint(0, len(self.enviroment), (batch_size,))
        return self.enviroment[indices], self.labels[indices]

