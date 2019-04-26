import dataset
import numpy as np
import torch
import train
from tqdm import tqdm
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

MODEL = "./models/model_9conv.pth"
OUTPUT_DIR = "./test/"


def calc_score(loader, net, device):
    correct = 0
    cnt = 0
    with torch.no_grad():
        net = net.eval()
        for images, labels, imfiles in loader:
            images = images.to(device)
            labels = labels.view(labels.size(0))
            output = net(images).cpu()
            comparison = np.argmax(output, axis=1) == labels
            correct += torch.sum(comparison)
            cnt += images.size(0)

            for idx, result in enumerate(comparison):
                if result.item() == 0:
                    print(imfiles[idx], "Expected: %s" % labels[idx].item(), "Pred: %s" % np.argmax(output[idx]).item())


    score = correct.item() / cnt
    print(score)
    return score


def main():
    state = torch.load(MODEL)
    net = train.Net()
    net.load_state_dict(state)

    data = dataset.TestDataset(start=2639, size=267)
    # data = dataset.TestDataset(start=2907, size=166)
    # data = dataset.TestDataset(start=3073, size=166)
    # data = dataset.TestDataset(start=2639, size=600)
    loader = DataLoader(data, batch_size=1)
    calc_score(loader, net, "cpu")

if __name__ == "__main__":
    main()
