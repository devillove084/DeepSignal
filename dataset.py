import torch
import torch.nn as nn
import torch.optim as optim
from torch._C import device, dtype
from torch.utils.data import Dataset
import os

from PIL import Image
from torch.utils.data.dataset import random_split
from torchvision import transforms as T
import numpy as np

from torch.utils.data import DataLoader


from lenet5 import LeNet5

class SignalData(Dataset):
    def __init__(self, root_dir):
        f = open(root_dir, "r")
        self.images_data = list()
        for lines in f.readlines():
            self.images_data.append(lines)

        self.transform = T.Compose([
            T.Resize((65, 65)),
            T.ToTensor()
            #T.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, index):
        images_dat = self.images_data[index]
        images_dir = images_dat.split(' ')[0]
        images_label = images_dat.split(' ')[1]
        #print(images_dir)
        img = Image.open(images_dir)
        img = img.convert("RGB")

        img = self.transform(img)
        #print(img.shape)
        return img.float(), np.int32(images_label)

train_dataset = SignalData("test.txt")
print(len(train_dataset))

train_n, test_n = random_split(train_dataset, [500, 200], generator=torch.Generator().manual_seed(42))


data_train = DataLoader(train_n, batch_size=4, shuffle=True, num_workers=4)

data_test = DataLoader(test_n, batch_size=4, shuffle=False, num_workers=4)



if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


net = LeNet5().to(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=2e-3)

cur_batch_win = None
cur_batch_win_opts = {
    'title': 'Epoch Loss Trace',
    'xlabel': 'Batch Number',
    'ylabel': 'Loss',
    'width': 1200,
    'height': 600,
}

def train(epoch):
    global cur_batch_win
    net.train()
    loss_list, batch_list = list(), list()
    for i, (image, labels) in enumerate(data_train):
        print(image.shape)

        optimizer.zero_grad()
        output = net(torch.Tensor(image).cuda())
        labels = torch.tensor(labels, dtype=torch.long)
        loss = criterion(output, labels.cuda())
        loss_list.append(loss.cpu().item())
        batch_list.append(i+1)

        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

        loss.backward()
        optimizer.step()

def test():
    net.eval()
    total_correct = 0
    avg_loss = 0.0

    for i, (images, labels) in enumerate(data_test):
        output = net(torch.Tensor(images).cuda())
        labels = torch.tensor(labels, dtype=torch.long)
        avg_loss += criterion(output, labels.cuda()).sum()
        pred = output.cpu().detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))
    print(float(total_correct))


def main():
    for i in range(1, 50):
        train(i)
        test()

main()
# for e in range(2):
#     for step, data in enumerate(tl):
#         data_input, label = data
#         #print(step, ":", label)
#         print(data_input)