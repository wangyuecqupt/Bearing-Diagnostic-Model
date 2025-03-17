# QCNNResNet  添加t-sne （relu在pool后）
# 6层二次卷积神经网络
# 旁边引入加5个二次卷积神经网络
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, flop_count_str
from torchsummary import summary
import torch.nn.functional as F

from Model.ConvQuadraticOperation import ConvQuadraticOperation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()


class ResNeStShortcut(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size=1, padding=0):
        super(ResNeStShortcut, self).__init__()
        self.conv = ConvQuadraticOperation(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class QCNNResNet(nn.Module):
    def __init__(self, ) -> object:
        super(QCNNResNet, self).__init__()
        self.cnn = nn.Sequential()

        self.conv1 = ConvQuadraticOperation(1, 16, 64, 8, 28)
        self.bn1 = nn.BatchNorm1d(16)
        self.maxpool1 = nn.MaxPool1d(2, 2)
        self.relu1 = nn.ReLU()  # ReLU moved after MAXPool

        self.resnet_shortcut1_2 = ConvQuadraticOperation(16, 32, 2, 2, 0)  # shortcut from layer1 to layer2
        self.bn1_2 = nn.BatchNorm1d(32)

        self.conv2 = ConvQuadraticOperation(16, 32, 3, 1, 1)
        self.bn2 = nn.BatchNorm1d(32)
        self.maxpool2 = nn.MaxPool1d(2, 2)
        self.relu2 = nn.ReLU()  # ReLU moved after MAXPool

        self.resnet_shortcut2_3 = ConvQuadraticOperation(32, 64, 2, 2, 0)  # shortcut from layer2 to layer3
        self.bn2_3 = nn.BatchNorm1d(64)

        self.conv3 = ConvQuadraticOperation(32, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.maxpool3 = nn.MaxPool1d(2, 2)
        self.relu3 = nn.ReLU()  # ReLU moved after MAXPool

        self.resnet_shortcut3_4 = ConvQuadraticOperation(64, 64, 2, 2, 0)  # shortcut from layer3 to layer4
        self.bn3_4 = nn.BatchNorm1d(64)

        self.conv4 = ConvQuadraticOperation(64, 64, 3, 1, 1)
        self.bn4 = nn.BatchNorm1d(64)
        self.maxpool4 = nn.MaxPool1d(2, 2)
        self.relu4 = nn.ReLU()  # ReLU moved after MAXPool

        self.resnet_shortcut4_5 = ConvQuadraticOperation(64, 64, 2, 2, 0)  # shortcut from layer4 to layer5
        self.bn4_5 = nn.BatchNorm1d(64)

        self.conv5 = ConvQuadraticOperation(64, 64, 3, 1, 1)
        self.bn5 = nn.BatchNorm1d(64)
        self.maxpool5 = nn.MaxPool1d(2, 2)
        self.relu5 = nn.ReLU()  # ReLU moved after MAXPool

        self.resnet_shortcut5_6 = ConvQuadraticOperation(64, 64, 4, 2, 0)  # shortcut from layer5 to layer6
        self.bn5_6 = nn.BatchNorm1d(64)

        self.conv6 = ConvQuadraticOperation(64, 64, 3, 1, 0)
        self.bn6 = nn.BatchNorm1d(64)
        self.maxpool6 = nn.MaxPool1d(2, 2)
        self.relu6 = nn.ReLU()  # ReLU moved after MAXPool

        self.fc1 = nn.Linear(192, 100)  # 64*3=192
        self.relu1_fc = nn.ReLU()
        self.dp = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 10)

    def get_features(self, x):
        out1 = self.relu1(self.maxpool1(self.bn1(self.conv1(x))))
        shortcut1_2 = self.bn1_2(self.resnet_shortcut1_2(out1))

        out2 = self.maxpool2(self.bn2(self.conv2(out1)))
        out2 = self.relu2((shortcut1_2 + out2))
        shortcut2_3 = self.bn2_3(self.resnet_shortcut2_3(out2))

        out3 = self.maxpool3(self.bn3(self.conv3(out2)))
        out3 = self.relu3((shortcut2_3 + out3))
        shortcut3_4 = self.bn3_4(self.resnet_shortcut3_4(out3))

        out4 = self.maxpool4(self.bn4(self.conv4(out3)))
        out4 = self.relu4((shortcut3_4 + out4))
        shortcut4_5 = self.bn4_5(self.resnet_shortcut4_5(out4))

        out5 = self.maxpool5(self.bn5(self.conv5(out4)))
        out5 = self.relu5((shortcut4_5 + out5))
        shortcut5_6 = self.bn5_6(self.resnet_shortcut5_6(out5))

        out6 = self.maxpool6(self.bn6(self.conv6(out5)))
        out6 = self.relu6((shortcut5_6 + out6))

        out = self.fc1(out6.view(out6.size(0), -1))
        out = self.relu1_fc(out)
        out = self.dp(out)
        out = self.fc2(out)

        return out.view(out.size(0), -1)  # out输出输出层，需要输出一维的数据

    def forward(self, x):
        out1 = self.relu1(self.maxpool1(self.bn1(self.conv1(x))))
        shortcut1_2 = self.bn1_2(self.resnet_shortcut1_2(out1))

        out2 = self.maxpool2(self.bn2(self.conv2(out1)))
        out2 = self.relu2((shortcut1_2 + out2))
        shortcut2_3 = self.bn2_3(self.resnet_shortcut2_3(out2))

        out3 = self.maxpool3(self.bn3(self.conv3(out2)))
        out3 = self.relu3((shortcut2_3 + out3))
        shortcut3_4 = self.bn3_4(self.resnet_shortcut3_4(out3))

        out4 = self.maxpool4(self.bn4(self.conv4(out3)))
        out4 = self.relu4((shortcut3_4 + out4))
        shortcut4_5 = self.bn4_5(self.resnet_shortcut4_5(out4))

        out5 = self.maxpool5(self.bn5(self.conv5(out4)))
        out5 = self.relu5((shortcut4_5 + out5))
        shortcut5_6 = self.bn5_6(self.resnet_shortcut5_6(out5))

        out6 = self.maxpool6(self.bn6(self.conv6(out5)))
        out6 = self.relu6((shortcut5_6 + out6))

        out = self.fc1(out6.view(out6.size(0), -1))
        out = self.relu1_fc(out)
        out = self.dp(out)
        out = self.fc2(out)
        return F.softmax(out, dim=1)


if __name__ == '__main__':
    X = torch.rand(1, 1, 2048).to(device)
    m = QCNNResNet().to(device)
    summary(m, (1, 2048))
    # print(flop_count_str(FlopCountAnalysis(m.cuda(), X)))  # Commented out since it requires fvcore installed
