import torch
from torch import nn


class Conv1d_bn(nn.Module):
    def __init__(self, in_filters, out_filters, filter_size, stride=1, padding=0):
        super(Conv1d_bn, self).__init__()
        self.conv = nn.Conv1d(in_filters, out_filters, filter_size,
                         stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_filters)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class GuidedLeakyReLU(torch.autograd.Function):
    def __init__(self, alpha):
        super(GuidedLeakyReLU, self).__init__()
        self.alpha = alpha

    def forward(self, input):
        self.save_for_backward(input)
        output = input.clamp(min=0.)
        output = output + self.alpha * (input.clamp(max=0.))
        return output

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clamp(min=0.)
        grad_input[input < 0.] = 0.
        return grad_input


class GuidedReLU(torch.autograd.Function):
    def forward(self, input):
        self.save_for_backward(input)
        output = input.clamp(min=0.)
        return output

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clamp(min=0)
        grad_input[input < 0] = 0
        return grad_input


class RCConv1d(torch.nn.Module):
    def __init__(self, in_filters, out_filters, width, sum_rc=False, use_bn=False):
        super(RCConv1d, self).__init__()
        self.w = torch.nn.Parameter(torch.randn((out_filters, in_filters, width)))
        self.b = None
        if not use_bn:
            self.b = torch.nn.Parameter(torch.zeros(out_filters))
        from torch.nn import init
        init.xavier_uniform(self.w)
        self.register_buffer('o_r_idx', torch.LongTensor(torch.from_numpy(np.arange(self.w.size(0) - 1, -1, -1))))
        self.register_buffer('i_r_idx', torch.LongTensor(torch.from_numpy(np.arange(self.w.size(1) - 1, -1, -1))))
        self.register_buffer('l_r_idx', torch.LongTensor(torch.from_numpy(np.arange(self.w.size(2) - 1, -1, -1))))
        self.sum_rc = sum_rc
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = torch.nn.BatchNorm1d(out_filters)

    def forward(self, x):
        o_r_idx = torch.autograd.Variable(self.o_r_idx)
        i_r_idx = torch.autograd.Variable(self.i_r_idx)
        l_r_idx = torch.autograd.Variable(self.l_r_idx)
        fw = F.conv1d(x, self.w, bias=self.b)

        rc = F.conv1d(x, self.w.index_select(2, l_r_idx).index_select(1, i_r_idx),
                      bias=self.b)
        if not self.sum_rc:
            if self.use_bn:
                length = fw.size(2)
                x = self.bn(torch.cat((fw, rc), dim=2))
                fw = x[:, :, :length]
                rc = x[:, :, length:]
            rc = rc.index_select(1, o_r_idx)
            return torch.cat((fw, rc), dim=1)
        else:
            return fw + rc


class RCWeightedSum(torch.nn.Module):
    def __init__(self, out_filters, length):
        super(RCWeightedSum, self).__init__()
        self.w = torch.nn.Parameter(torch.randn((out_filters, 1, length)))
        from torch.nn import init
        init.xavier_uniform(self.w)

        self.register_buffer('o_r_idx', torch.LongTensor(torch.from_numpy(np.arange(self.w.size(0) - 1, -1, -1))))
        self.register_buffer('l_r_idx', torch.LongTensor(torch.from_numpy(np.arange(self.w.size(2) - 1, -1, -1))))

    def forward(self, x):
        assert x.size(2) == self.w.size(2), "Not matching length"
        o_r_idx = torch.autograd.Variable(self.o_r_idx)
        l_r_idx = torch.autograd.Variable(self.l_r_idx)
        half = x.size(1) // 2
        fw = F.conv1d(x[:, :half, :], self.w, groups=half)
        w_rc = self.w.index_select(2, l_r_idx).index_select(0, o_r_idx)
        rc = F.conv1d(x[:, half:, :], w_rc, groups=half)
        x = torch.cat((fw, rc), dim=1)
        x = torch.squeeze(x)
        return x