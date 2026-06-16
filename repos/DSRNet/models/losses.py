import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vgg import Vgg19


def compute_grad(img):
    gradx = img[..., 1:, :] - img[..., :-1, :]
    grady = img[..., 1:] - img[..., :-1]
    return gradx, grady


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, predict, target):
        predict_gradx, predict_grady = compute_grad(predict)
        target_gradx, target_grady = compute_grad(target)

        return self.loss(predict_gradx, target_gradx) + self.loss(predict_grady, target_grady)


class MultipleLoss(nn.Module):
    def __init__(self, losses, weight=None):
        super(MultipleLoss, self).__init__()
        self.losses = nn.ModuleList(losses)
        self.weight = weight or [1 / len(self.losses)] * len(self.losses)

    def forward(self, predict, target):
        total_loss = 0
        for weight, loss in zip(self.weight, self.losses):
            total_loss += loss(predict, target) * weight
        return total_loss


def _blur_depthwise(image, radius):
    channels = image.shape[1]
    kernel = image.new_tensor([
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]).view(1, 1, 3, 3)
    kernel = kernel.repeat(channels, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode='replicate')
    return F.conv2d(image, kernel, groups=channels, dilation=radius)


def frequency_decomposition(image, levels=3):
    levels = max(int(levels), 1)
    high_freq = torch.zeros_like(image)
    low_freq = image
    for i in range(levels):
        radius = 2 ** i
        next_low = _blur_depthwise(low_freq, radius)
        high_freq = high_freq + (low_freq - next_low)
        low_freq = next_low
    return high_freq, low_freq


def normalize_map(weight, eps=1e-6):
    dims = tuple(range(2, weight.dim()))
    min_v = weight.amin(dim=dims, keepdim=True)
    max_v = weight.amax(dim=dims, keepdim=True)
    return (weight - min_v) / (max_v - min_v + eps)


class FrequencyLoss(nn.Module):
    def __init__(self, levels=3):
        super().__init__()
        self.levels = levels
        self.criterion = nn.L1Loss()

    def forward(self, predict, target):
        pred_hf, pred_lf = frequency_decomposition(predict, self.levels)
        target_hf, target_lf = frequency_decomposition(target, self.levels)
        return self.criterion(pred_lf, target_lf) + self.criterion(pred_hf, target_hf)


class PriorWeightedLoss(nn.Module):
    def __init__(self, levels=3):
        super().__init__()
        self.levels = levels

    def _prior_map(self, input_i, target_t, target_r, prior_source):
        if prior_source == 'input_minus_t':
            prior_base = input_i - target_t
        elif prior_source == 'target_r':
            prior_base = target_r
        else:
            raise ValueError('Unsupported prior_source: {}'.format(prior_source))
        prior = torch.mean(torch.abs(prior_base.detach()), dim=1, keepdim=True)
        return normalize_map(prior)

    def _hf_map(self, input_i):
        high_freq, _ = frequency_decomposition(input_i.detach(), self.levels)
        high_freq = torch.mean(torch.abs(high_freq), dim=1, keepdim=True)
        return normalize_map(high_freq)

    def weight_map(self, input_i, target_t, target_r, prior_source='target_r'):
        prior = self._prior_map(input_i, target_t, target_r, prior_source)
        high_freq = self._hf_map(input_i)
        return torch.clamp(1.0 + prior * high_freq, 1.0, 2.0)

    def forward(self, out_t, target_t, out_r, target_r, input_i, prior_source='target_r'):
        weight = self.weight_map(input_i, target_t, target_r, prior_source)
        residual = torch.abs(out_t - target_t) + torch.abs(out_r - target_r)
        return (weight * residual).mean()


class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False


class VGGLoss(nn.Module):
    def __init__(self, vgg=None, weights=None, indices=None, normalize=True):
        super(VGGLoss, self).__init__()
        if vgg is None:
            self.vgg = Vgg19().cuda()
        else:
            self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10 / 1.5]
        self.indices = indices or [2, 7, 12, 21, 30]
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        else:
            self.normalize = None

    def forward(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

        return loss


class ReconsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, out_t, out_r, out_rr, input_i):
        content_diff = self.criterion(out_t + out_r + out_rr, input_i)
        return content_diff


class ExclusionLoss(nn.Module):
    def __init__(self, level=3, eps=1e-6):
        super().__init__()
        self.level = level
        self.eps = eps

    def forward(self, img_T, img_R):
        grad_x_loss = []
        grad_y_loss = []

        for l in range(self.level):
            grad_x_T, grad_y_T = compute_grad(img_T)
            grad_x_R, grad_y_R = compute_grad(img_R)

            alphax = (2.0 * torch.mean(torch.abs(grad_x_T))) / (torch.mean(torch.abs(grad_x_R)) + self.eps)
            alphay = (2.0 * torch.mean(torch.abs(grad_y_T))) / (torch.mean(torch.abs(grad_y_R)) + self.eps)

            gradx1_s = (torch.sigmoid(grad_x_T) * 2) - 1  # mul 2 minus 1 is to change sigmoid into tanh
            grady1_s = (torch.sigmoid(grad_y_T) * 2) - 1
            gradx2_s = (torch.sigmoid(grad_x_R * alphax) * 2) - 1
            grady2_s = (torch.sigmoid(grad_y_R * alphay) * 2) - 1

            grad_x_loss.append(((torch.mean(torch.mul(gradx1_s.pow(2), gradx2_s.pow(2)))) + self.eps) ** 0.25)
            grad_y_loss.append(((torch.mean(torch.mul(grady1_s.pow(2), grady2_s.pow(2)))) + self.eps) ** 0.25)

            img_T = F.interpolate(img_T, scale_factor=0.5, mode='bilinear')
            img_R = F.interpolate(img_R, scale_factor=0.5, mode='bilinear')
        loss_gradxy = torch.sum(sum(grad_x_loss) / 3) + torch.sum(sum(grad_y_loss) / 3)

        return loss_gradxy / 2


def init_loss(opt):
    loss_dic = {}
    pixel_loss = MultipleLoss([nn.MSELoss(), GradientLoss()], [0.3, 0.6])
    loss_dic['t_pixel'] = pixel_loss
    loss_dic['r_pixel'] = pixel_loss
    loss_dic['recons'] = ReconsLoss()
    loss_dic['exclu'] = ExclusionLoss(level=3)
    loss_dic['freq'] = FrequencyLoss(levels=getattr(opt, 'freq_levels', 3))
    loss_dic['prior'] = PriorWeightedLoss(levels=getattr(opt, 'freq_levels', 3))
    return loss_dic


if __name__ == '__main__':
    x = torch.randn(3, 32, 224, 224).cuda()
