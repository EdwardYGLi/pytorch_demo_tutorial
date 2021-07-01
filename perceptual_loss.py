"""
Created by edwardli on 6/30/21
"""
import torch
import torchvision.models.vgg as vgg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Normalization(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class PerceptualLoss(torch.nn.Module):
    """
    Evaluates how closely two images match by comparing a feature map.
    """

    def __init__(self):
        super().__init__()
        self.vgg = vgg.vgg13_bn(pretrained=True)

        # freeze all model parameters
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.normalization = Normalization(
            normalization_mean, normalization_std)

    def _extract_features(self, x):
        """
         Extract features for a given image
         :param x: input
         :return: feature dictionary
         """
        x = self.normalization(x)

        for i in range(12):
            x = self.vgg.features[i](x)
        return x

    def forward(self, view_a, view_b):
        """
        Calculates the mid level feature loss.
        :param view_a: image A
        :param view_b: image B
        :return: perceptual loss
        """
        if view_a.shape != view_b.shape:
            raise Exception(
                "Input images to perceptual loss have mismatching shape")

        if view_a.shape[1] == 1:
            # Duplicate number of channels from 1 to 3 (grayscale to RGB
            # representation)
            view_a = torch.cat((view_a, view_a, view_a), 1)
            view_b = torch.cat((view_b, view_b, view_b), 1)

        p_loss = torch.mean(
            torch.abs(
                self._extract_features(view_a) -
                self._extract_features(view_b)))

        return p_loss


_perceptual_loss_instance = None


def perceptual_loss(view_a, view_b):
    """
    Functional wrapper for perceptual loss.
    """
    global _perceptual_loss_instance
    if _perceptual_loss_instance is None:
        _perceptual_loss_instance = PerceptualLoss().to(device)
    return _perceptual_loss_instance(view_a, view_b)