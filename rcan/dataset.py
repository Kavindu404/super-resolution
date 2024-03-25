from torch.utils.data import Dataset


class SRDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx], self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample

