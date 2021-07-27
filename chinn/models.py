import torch
from torch import nn
from chinn.layers import *
import torch.nn.functional as F


class ReLUMixin():
    def _relu(self, inp):
        leaky_alpha = 1. / 5.5
        if self.guided and self.leaky:
            return GuidedLeakyReLU(leaky_alpha)(inp)
        elif self.guided:
            return GuidedReLU()(inp)
        elif self.leaky:
            return F.leaky_relu(inp, leaky_alpha)
        else:
            return F.relu(inp)


class NNClassifier(torch.nn.Module, ReLUMixin):
    def __init__(self, input_dim=206, leaky=True, guided=False, legacy=False):
        super(NNClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 1024)
        if legacy:
            self.fc = torch.nn.Linear(1024,2)
        else:
            self.fc = torch.nn.Linear(1024, 1)
        self.softmax = torch.nn.Softmax()
        self.drop = torch.nn.Dropout(0.4)

        self.sigmoid = torch.nn.Sigmoid()

        self.leaky = leaky
        self.guided = guided


    def forward(self, x):
        x = self.drop(x)
        x = self.fc1(x)
        x = self._relu(x)
        x = self.drop(x)
        x = self.fc(x)
        x = torch.squeeze(x, 1)
        return x


class NNClassifierBN(torch.nn.Module, ReLUMixin):
    def __init__(self, input_dim=206, leaky=True, guided=False, use_bn=False):
        super(NNClassifierBN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 1024)
        self.fc = torch.nn.Linear(1024, 2)
        self.softmax = torch.nn.Softmax()
        self.drop = torch.nn.Dropout(0.4)

        self.sigmoid = torch.nn.Sigmoid()

        self.leaky = leaky
        self.guided = guided
        self.use_bn = use_bn
        if use_bn:
            self.bn1 = torch.nn.BatchNorm1d(1024, momentum=0.1)


    def forward(self, x):
        x = self.fc1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self._relu(x)
        x = self.fc(x)
        return x



class RCAnchorModel(torch.nn.Module):
    def __init__(self, in_filters=4):
        super(RCAnchorModel, self).__init__()
        fold = 2
        sum_rc = False
        use_bn = True
        self.conv1 = RCConv1d(in_filters, 240, 8, sum_rc=sum_rc, use_bn=use_bn)
        self.conv2 = RCConv1d(240 * fold, 320, 8, sum_rc=sum_rc, use_bn=use_bn)
        self.conv3 = RCConv1d(320 * fold, 480, 8, sum_rc=sum_rc, use_bn=use_bn)
        self.weighted_sum = RCWeightedSum(480, 53)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=4, stride=4)
        self.drop2 = torch.nn.Dropout(0.2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.drop2(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.drop2(x)
        x = self.relu(self.conv3(x))
        x = self.weighted_sum(x)
        return x


class PartialDeepSeaModel(torch.nn.Module, ReLUMixin):
    def __init__(self, in_filters=4, use_weightsum=False, leaky=False,
                 use_bn=False, use_fc=False, guided=False, use_sigmoid=False):
        super(PartialDeepSeaModel, self).__init__()
        self.use_sigmoid = use_sigmoid
        self._set_filters()
        self._build_model(in_filters, use_weightsum, leaky, use_bn, use_fc, guided)

    def _set_filters(self):
        self.num_filters = [128, 256, 128]

    def _build_model(self, in_filters=4, use_weightsum=False, leaky=False, use_bn=False, use_fc=False, guided=False):
        if not use_bn:
            self.conv1 = nn.Conv1d(in_filters, self.num_filters[0], 8)
            self.conv2 = nn.Conv1d(self.num_filters[0], self.num_filters[1], 8)
            self.conv3 = nn.Conv1d(self.num_filters[1], self.num_filters[2], 8)
        else:
            self.conv1 = Conv1d_bn(in_filters, self.num_filters[0], 8)
            self.conv2 = Conv1d_bn(self.num_filters[0], self.num_filters[1], 8)
            self.conv3 = Conv1d_bn(self.num_filters[1], self.num_filters[2], 8)

        if use_weightsum:
            self.weighted_sum = torch.nn.Parameter(torch.randn((self.num_filters[2], 53)))
            torch.nn.init.kaiming_uniform_(self.weighted_sum)
        elif use_fc:
            self.fc1 = nn.Linear(self.num_filters[2] * 53, 512)
            self.fc = nn.Linear(512, self.num_filters[2])
        else:
            if not use_bn:
                self.conv4 = nn.Conv1d(self.num_filters[2], self.num_filters[2], 13)
            else:
                self.conv4 = Conv1d_bn(self.num_filters[2], self.num_filters[2], 13)

        self.use_weightsum = use_weightsum
        self.use_fc = use_fc

        self.drop2 = nn.Dropout(0.5)

        self.maxpool = nn.MaxPool1d(kernel_size=4, stride=4)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.guided = guided
        self.leaky = leaky
        self.use_bn = use_bn


    def forward(self, x):
        x = self._relu(self.conv1(x))
        x = self.maxpool(x)
        if not self.use_bn:
            x = self.drop2(x)
        x = self._relu(self.conv2(x))
        x = self.maxpool(x)
        if not self.use_bn:
            x = self.drop2(x)
        x = self._relu(self.conv3(x))
        if not self.use_bn:
            x = self.drop2(x)
        if self.use_weightsum:
            x = torch.sum(torch.mul(x, self.weighted_sum), dim=2, keepdim=False)
        elif self.use_fc:
            x = x.view((-1, self.num_filters[2] * 53))
            x = self._relu(self.fc1(x))
            x = self.drop2(x)
            x = self._relu(self.fc(x))
        else:
            x = self.maxpool(x)
            x = self.tanh(self.conv4(x))
            x = torch.squeeze(x, 2)
        if self.use_weightsum:
            if self.use_sigmoid:
                x = self.sigmoid(x)
            else:
                x = self.tanh(x)
        return x

    def get_weightsum(self, x):
        x = self._relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.drop2(x)
        x = self._relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.drop2(x)
        x = self._relu(self.conv3(x))
        x = self.drop2(x)
        x = torch.sum(torch.mul(x, self.weighted_sum), dim=2, keepdim=False)
        return x

    def get_conv_activations(self, x, level):
        assert level <= 3
        convs = [self.conv1, self.conv2, self.conv3]
        for i in range(level):
            x = self._relu(convs[i](x))
            if i < level - 1:
                x = self.maxpool(x)
        return x


class LargePartialDeepSeaModel(PartialDeepSeaModel):
    def _set_filters(self):
        self.num_filters = [256, 320, 256]
