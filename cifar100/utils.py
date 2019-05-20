import torch

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1).cuda()
        self.bn1 = torch.nn.BatchNorm2d(16).cuda()
        self.conv11 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1).cuda()
        self.bn11 = torch.nn.BatchNorm2d(16).cuda()
        self.pool1 = torch.nn.MaxPool2d(3, stride=2, padding=1).cuda()

        self.conv2 = torch.nn.Conv2d(16, 32, 3, stride=1, padding=1).cuda()
        self.bn2 = torch.nn.BatchNorm2d(32).cuda()
        self.conv22 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1).cuda()
        self.bn22 = torch.nn.BatchNorm2d(32).cuda()
        self.pool2 = torch.nn.MaxPool2d(3, stride=2, padding=1).cuda()

        self.conv3 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1).cuda()
        self.bn3 = torch.nn.BatchNorm2d(64).cuda()
        self.conv33 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1).cuda()
        self.bn33 = torch.nn.BatchNorm2d(64).cuda()
        self.pool3 = torch.nn.MaxPool2d(3, stride=2, padding=1).cuda()

        self.conv4 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=1).cuda()
        self.bn4 = torch.nn.BatchNorm2d(128).cuda()
        self.conv44 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=1).cuda()
        self.bn44 = torch.nn.BatchNorm2d(128).cuda()
        self.pool = torch.nn.AdaptiveAvgPool2d(1).cuda()


        self.fc1 = torch.nn.Linear(128, 20).cuda()
        self.fc2 = torch.nn.Linear(128, 100).cuda()
        self.tanh = torch.nn.Tanh().cuda()

    def forward(self, x):
        l = self.tanh(self.bn1(self.conv1(x)))
        l = l + self.tanh(self.bn11(self.conv11(l)))
        l = self.pool1(l)

        l = self.tanh(self.bn2(self.conv2(l)))
        l = l + self.tanh(self.bn22(self.conv22(l)))
        l = self.pool2(l)

        l = self.tanh(self.bn3(self.conv3(l)))
        l = l + self.tanh(self.bn33(self.conv33(l)))
        l = self.pool3(l)

        l = self.tanh(self.bn4(self.conv4(l)))
        l = l + self.tanh(self.bn44(self.conv44(l)))
        l = self.pool(l).view(-1, 128)

        pred_coarse = self.fc1(l)
        pred_fine = self.fc2(l)
        return pred_coarse, pred_fine

if __name__ == '__main__':
    net = Network()
    from IPython import embed
    embed()

