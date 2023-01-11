from turtle import forward
from typing import List, Literal, Optional, Sequence, Tuple

from einops import reduce, rearrange
import numpy as np 
import torch
from torch import nn
from torch._C import Size
from torch.functional import Tensor
from torch.nn import BatchNorm2d, Conv2d, ConvTranspose2d, ReLU, Softmax, Sigmoid
from torch.nn import functional as F
from torch.nn.init import kaiming_normal_
from torch.nn.modules.pooling import MaxPool2d
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode


class VanilaSegmentor(nn.Module):

    def __init__(
        self, input_shape: Size, num_classes: int, num_filters: int, enable_batchnorm: bool,
        upsampling_mode: Literal['deconv', 'nn'] = 'deconv', num_attention_classes: Optional[int] = None,
        enable_attention_gates: bool = True,
        ):
        """Vanila Segmentor.
        Original implementation by Gabriele Valvano.

        Vanila U-Net with Adversarial Attention Gate module.

        params:
        input_shape: torch.Size         input size sample.
        num_classes: int                Number of classes. For binary task, 1 for grayscale prediction, 2 for one-hot.
        enable_batchnorm: bool          Enabling batchnorm.
        upsampling_mode: str            Upsample method. 'deconv' for deconvolution. 'nn' for nearest neighbor interpolation.
        num_attention_classes: str      Number of classes for attention gating. Default is equal to num_classes.
        """
        super().__init__()
        self.input_shape = input_shape
        assert input_shape[2] == input_shape[3], 'Only support square input.'
        in_channels = int(input_shape[1])
        self.num_classes = num_classes
        if num_attention_classes is None:
            self.attention_classes = num_classes
        else:
            self.attention_classes = num_attention_classes
        self.num_filters = num_filters
        self.enable_bn = enable_batchnorm
        self.upsampling_mode = upsampling_mode
        self.enable_att = enable_attention_gates

        # Encoder stack
        self.pool_0, self.enc_0 = self._encoder_block(
            in_channels, num_filters, enable_batchnorm)
        self.pool_1, self.enc_1 = self._encoder_block(
            num_filters, 2 * num_filters, enable_batchnorm)
        self.pool_2, self.enc_2 = self._encoder_block(
            2 * num_filters, 4 * num_filters, enable_batchnorm)
        self.pool_3, self.enc_3 = self._encoder_block(
            4 * num_filters, 8 * num_filters, enable_batchnorm)

        # Bottleneck
        self.bottleneck = self._bottleneck_block(
            num_filters * 8, enable_batchnorm)

        # Decoder
        self.up_3, self.dec_3, self.att_3 = self._decoder_block(
            input_shape, num_filters * 8, num_filters * 8, upsampling_mode, num_filters * 8, self.attention_classes, enable_batchnorm)
        self.up_2, self.dec_2, self.att_2 = self._decoder_block(
            input_shape, num_filters * 8, num_filters * 4, upsampling_mode, num_filters * 4, self.attention_classes, enable_batchnorm)
        self.up_1, self.dec_1, self.att_1 = self._decoder_block(
            input_shape, num_filters * 4, num_filters * 2, upsampling_mode, num_filters * 2, self.attention_classes, enable_batchnorm)
        self.up_0, self.dec_0, self.att_0 = self._decoder_block(
            input_shape, num_filters * 2, num_filters, upsampling_mode, num_filters, self.attention_classes, enable_batchnorm)

        # FC
        self.fc = Conv2d(num_filters, num_classes, kernel_size=1, stride=1)
        kaiming_normal_(self.fc.weight, nonlinearity='relu')

    @staticmethod
    def _encoder_block(
        in_channels: int, num_filters: int, enable_batchnorm: bool):
        module = []
        conv_0 = Conv2d(
            in_channels=in_channels,
            out_channels=num_filters,
            kernel_size=(3, 3),
            stride=1,
            padding=1
            )
        kaiming_normal_(conv_0.weight, nonlinearity='relu')
        module.append(conv_0)
        if enable_batchnorm:
            bn_0 = BatchNorm2d(num_features=num_filters)
            module.append(bn_0)
        module.append(ReLU(inplace=True))
        conv_1 = Conv2d(
            in_channels=num_filters, out_channels=num_filters,
            kernel_size=(3, 3),
            stride=1,
            padding=1 
        )
        kaiming_normal_(conv_1.weight, nonlinearity='relu')
        module.append(conv_1)
        if enable_batchnorm:
            bn_1 = BatchNorm2d(num_features=num_filters)
            module.append(bn_1)
        module.append(ReLU(inplace=True))
        # Calculate padding (stride = 2, filter 2 x 2, dilation=1)
        # pad = ZeroPad2d(get_same_padding_conv(input_size, 2, 2))
        pooling = MaxPool2d(
            kernel_size=(2,2),
            stride=2,
            )
        # pooling = nn.Sequential(pad, pooling)
        encoder_stack = nn.Sequential(*module)
        return pooling, encoder_stack

    @staticmethod
    def _bottleneck_block(num_filters, enable_batchnorm):
        module = []
        conv_0 = Conv2d(
            num_filters,
            num_filters,
            kernel_size=1,
            stride=1
        )
        kaiming_normal_(conv_0.weight, nonlinearity='relu')
        if enable_batchnorm:
            bn_0 = BatchNorm2d(num_filters)
            module.append(bn_0)
        module.append(ReLU(inplace=True))
        conv_1 = Conv2d(
            num_filters,
            num_filters,
            kernel_size=1,
            stride=1,
            )
        kaiming_normal_(conv_1.weight, nonlinearity='relu')
        module.append(conv_1)
        if enable_batchnorm:
            bn_1 = BatchNorm2d(num_filters)
            module.append(bn_1)
        module.append(ReLU(inplace=True))
        bottleneck_stack = nn.Sequential(*module)
        return bottleneck_stack

    @staticmethod
    def _decoder_block(
        input_shape: Size,
        prev_layers_in: int,
        skip_layer_in: int,
        upsampling_mode: Literal['deconv', 'nn'],
        num_filters: int,
        num_classes: int,
        enable_batchnorm: bool):
        """Upsampling by Nearest-Neighbor Interpolation.
        """
        modules = []
        input_size = int(input_shape[3])
        in_channels = int(input_shape[1])
        new_shape = int(2.0 * input_size)
        # Deconvolution input channels
        if upsampling_mode == 'deconv':
            upsampler = ConvTranspose2d(
                in_channels=prev_layers_in,
                out_channels=num_filters,
                kernel_size=2,
                stride=2,
                )
            kaiming_normal_(upsampler.weight, nonlinearity='relu')
        elif upsampling_mode == 'nn':
            scale_up = Resize(
                size=(new_shape, new_shape),
                interpolation=InterpolationMode.NEAREST
            )
            conv = Conv2d(
                in_channels=in_channels,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1
            )
            kaiming_normal_(conv.weight, nonlinearity='relu')
            upsampler = nn.Sequential(scale_up, conv) 
        else:
            NotImplementedError(f'Mode {upsampling_mode} not supported.')

        concat_layers = num_filters + skip_layer_in
        conv_1 = Conv2d(
            in_channels=concat_layers,
            out_channels=num_filters,
            kernel_size=3,
            stride=1,
            padding=1
            )
        kaiming_normal_(conv_1.weight, nonlinearity='relu')
        modules.append(conv_1)
        if enable_batchnorm:
            bn_1 = BatchNorm2d(num_filters)
            modules.append(bn_1)
        modules.append(ReLU(inplace=True))

        conv_2 = Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            stride=1,
            padding=1
        )
        kaiming_normal_(conv_2.weight, nonlinearity='relu')
        modules.append(conv_2)
        if enable_batchnorm:
            bn_2 = BatchNorm2d(num_features=num_filters)
            modules.append(bn_2)
        modules.append(ReLU(inplace=True))

        # Self-Attention
        attention_gate = AdversarialAttentionGate(num_filters, num_classes)

        return upsampler, nn.Sequential(*modules), attention_gate

    def forward(self, x: Tensor) -> Tuple[Optional[Tuple[Tensor, Tensor, Tensor, Tensor]], Tensor]:
        # Forwarding
        # Encoder Stack
        x_0 = self.enc_0(x)         # - - > Skip
        x_0_p = self.pool_0(x_0)    # v     Below

        x_1 = self.enc_1(x_0_p)     # - - > Skip
        x_1_p = self.pool_1(x_1)    # v     Below

        x_2 = self.enc_2(x_1_p)     # - - > Skip
        x_2_p = self.pool_2(x_2)    # v     Below

        x_3 = self.enc_3(x_2_p)     # - - > Skip
        x_3_p = self.pool_3(x_3)    # v     Below

        # Bottleneck, no pooling.
        x_b = self.bottleneck(x_3_p)
        # Decoder Stack
        d_3 = self.up_3(x_b)
        d_3 = torch.cat((x_3, d_3), dim=1)
        d_3 = self.dec_3(d_3)
        if self.enable_att:
            d_3, y_3 = self.att_3(d_3)

        d_2 = self.up_2(d_3)
        d_2 = torch.cat((x_2, d_2), dim=1) # 6 num_filters
        d_2 = self.dec_2(d_2)
        if self.enable_att:
            d_2, y_2 = self.att_2(d_2)

        d_1 = self.up_1(d_2)
        d_1 = torch.cat((x_1, d_1), dim=1)
        d_1 = self.dec_1(d_1)
        if self.enable_att:
            d_1, y_1 = self.att_1(d_1)

        d_0 = self.up_0(d_1)
        d_0 = torch.cat((x_0, d_0), dim=1)
        d_0 = self.dec_0(d_0)
        if self.enable_att:
            d_0, y_0 = self.att_0(d_0)

        agg_map = self.fc(d_0)

        if self.enable_att:
            return (y_0, y_1, y_2, y_3), agg_map
        else:
            return None, agg_map

    def forward_viz(self, x: Tensor):
        _, agg_map = self.forward(x)
        return agg_map

    def predict(self, x: Tensor, method: Literal['softmax', 'one-hot', 'origin', 'sigmoid'] = 'softmax'):
        """Segmentation on input x.
        params:
        x: Tensor       Raw image input tensor.
        method: str     Prediction output mode. ['softmax', 'one-hot', 'origin'].
                        Beware that 'one-hot' method is not differentiable.

        """
        att, agg_map = self.forward(x)
        if method == 'softmax':
            prediction = Softmax(dim=1)(agg_map)
        elif method == 'sigmoid':
            prediction = Sigmoid()(agg_map)
        elif method == 'one-hot':
            prediction = rearrange(F.one_hot(torch.argmax(agg_map, dim=1)), 'b h w c -> b c h w')
        elif method == 'origin':
            prediction = agg_map
        else:
            raise NotImplementedError
        return att, prediction


class AdversarialAttentionGate(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        """Adversarial Attention Gate Module

        Adversarial Attention Gate accept input from base residual module and perform
        dense convolution on feature maps with softmax to produce layers of segment and take hadamard product.

        params:

        in_channels     Number of input channels.
        out_channels    Number of output channels with each channel corresponding to each class segmentation.

        outputs:
        masked_x        Hadamard product of x and attention map.
        y_hat           Segmentor predicate.
        """
        super().__init__()
        self.conv1 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
        )
        self.softmax = Softmax(dim=1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # -> Masked input, predicate 
        x_prime = self.conv1(x)
        y_hat = self.softmax(x_prime)
        # Collapse y_hat, treat layer 0 as a background layer.
        attention_mask = reduce(y_hat[:, 1 if y_hat.ndim > 1 else 0:, :, :], 'b c h w -> b 1 h w', 'sum')
        # Masking input with attention map
        masked_x = x * attention_mask
        return masked_x, y_hat


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    NOTE: Just a tensor clipping function
    """
    t = t.float()
    # t_min=t_min.float()
    # t_max=t_max.float()

    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def create_mapping_kernel(kernel_size=7):
    """Mapping the kernel, creating k-square filter of the 1 x k x k kernel map.
    """
    # [kernel_size * kernel_size, kernel_size, kernel_size]
    # Generate kernel tensor
    kernel_arr = np.zeros((kernel_size * kernel_size, kernel_size, kernel_size), np.float32)
    # Initialize  of kernel map with 1
    for h in range(kernel_arr.shape[1]):
        for w in range(kernel_arr.shape[2]):
            # Sliding 1 along axis 1 with wrapping.
            kernel_arr[h * kernel_arr.shape[2] + w, h, w] = 1.0

    # [kernel_size * kernel_size, 1, kernel_size, kernel_size]
    kernel_tensor = torch.from_numpy(np.expand_dims(kernel_arr, axis=1))
    kernel_params = nn.parameter.Parameter(data=kernel_tensor.contiguous(), requires_grad=False)
    print(kernel_params.type())

    return kernel_params

def create_conv_kernel(in_channels, out_channels, kernel_size=3, avg=0.0, std=0.1):
    # [out_channels, in_channels, kernel_size, kernel_size]
    # Creating random valued conv kernel map, mostly a standard procedure with customable value distribution
    kernel_arr = np.random.normal(loc=avg, scale=std, size=(out_channels, in_channels, kernel_size, kernel_size))
    kernel_arr = kernel_arr.astype(np.float32)
    kernel_tensor = torch.from_numpy(kernel_arr)
    kernel_params = nn.parameter.Parameter(data=kernel_tensor.contiguous(), requires_grad=True)
    print(kernel_params.type())
    return kernel_params

def create_conv_bias(channels):
    # [channels, ]
    bias_arr = np.zeros(channels, np.float32)
    assert bias_arr.shape[0] % 2 == 1

    bias_arr[bias_arr.shape[0] // 2] = 1.0
    bias_tensor = torch.from_numpy(bias_arr)
    bias_params = nn.parameter.Parameter(data=bias_tensor.contiguous(), requires_grad=True)

    return bias_params

class basePC(nn.Module):
    def __init__(self, channels=256, pn_size=5, kernel_size=3, avg=0.0, std=0.1):
        """
        :param channels: the basic channels of feature maps.
        :param pn_size: the size of propagation neighbors.
        :param kernel_size: the size of kernel.
        :param avg: the mean of normal initialization.
        :param std: the standard deviation of normal initialization.
        
        Documentation:
        this is a propagation basis class.
        """
        super().__init__()
        self.kernel_size = kernel_size

        self.conv1_kernel = create_conv_kernel(in_channels=3, out_channels=channels,
                                               kernel_size=self.kernel_size, avg=avg, std=std)
        # conv1 -> conv4 propagation kernel
        self.conv4_kernel = create_conv_kernel(in_channels=channels, out_channels=2*channels,
                                               kernel_size=self.kernel_size, avg=avg, std=std)
        # conv4 -> conv 7 propagation kernel
        self.conv7_kernel = create_conv_kernel(in_channels=2*channels, out_channels=pn_size*pn_size,
                                               kernel_size=self.kernel_size, avg=avg, std=std)
        self.conv7_bias = create_conv_bias(pn_size*pn_size)
        
        
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn4 = nn.BatchNorm2d(2*channels)
        self.bn7 = nn.BatchNorm2d(pn_size*pn_size)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, input_src, input_thick, input_thin):
        input_all = torch.cat((input_src, input_thick, input_thin), dim=1)  # [b, 3, h, w] ##
        try:
            assert input_all.size()[1] == 3  # ##
        except AssertionError:
            raise ValueError(f'Expect concatenated confidence map to have channel depth of 3. Got {input_all.shape} instead.')
        # Full padding 
        fm_1 = F.conv2d(input_all, self.conv1_kernel, padding=self.kernel_size//2)
        fm_1 = self.bn1(fm_1)
        fm_1 = self.relu(fm_1)
        
        fm_4 = F.conv2d(fm_1, self.conv4_kernel, padding=self.kernel_size//2)
        fm_4 = self.bn4(fm_4)
        fm_4 = self.relu(fm_4)
        
        fm_7 = F.conv2d(fm_4, self.conv7_kernel, self.conv7_bias, padding=self.kernel_size//2)
        fm_7 = self.bn7(fm_7)
        fm_7 = F.relu(fm_7)

        return F.softmax(fm_7, dim=1)  # [b, pn_size * pn_size, h, w]

class adaptive_aggregationPC(nn.Module):
    def __init__(self, pn_size=5):
        """
        :param pn_size: the size of propagation neighbors.
        
        Aggregating confidence map with agg.coeff.
        """
        super().__init__()
        self.kernel_size = pn_size
        self.weight = create_mapping_kernel(kernel_size=self.kernel_size)

    def forward(self, input_thick, input_thin, agg_coeff):
        # agg.coeff is gained from progating 
        assert input_thick.size()[1] == 1 and input_thin.size()[1] == 1
        # Just maximize from both map.
        input_sal = torch.max(input_thick, input_thin)
        # Mapping...
        map_sal = F.conv2d(input_sal, self.weight, padding=self.kernel_size//2)
        # map_sal_inv = 1.0 - map_sal
        assert agg_coeff.size() == map_sal.size()

        # Calculate each pixels product.
        prod_sal = torch.sum(map_sal * agg_coeff, dim=1).unsqueeze(1)
        # prod_sal = F.sigmoid(prod_sal)
        # prod_sal_inv = torch.sum(map_sal_inv * agg_coeff, dim=1).unsqueeze(1)

        return prod_sal # [b, 1, h, w]


class baseC(nn.Module):
    def __init__(self, channels=256, pn_size=5, kernel_size=3, avg=0.0, std=0.1):
        """
        Base for single branch coarse stage
        :param channels: the basic channels of feature maps.
        :param pn_size: the size of propagation neighbors.
        :param kernel_size: the size of kernel.
        :param avg: the mean of normal initialization.
        :param std: the standard deviation of normal initialization.
        
        Documentation:
        this is a propagation basis class.
        """
        super().__init__()
        self.kernel_size = kernel_size

        self.conv1_kernel = create_conv_kernel(in_channels=2, out_channels=channels,
                                               kernel_size=self.kernel_size, avg=avg, std=std)
        # conv1 -> conv4 propagation kernel
        self.conv4_kernel = create_conv_kernel(in_channels=channels, out_channels=2*channels,
                                               kernel_size=self.kernel_size, avg=avg, std=std)
        # conv4 -> conv 7 propagation kernel
        self.conv7_kernel = create_conv_kernel(in_channels=2*channels, out_channels=pn_size*pn_size,
                                               kernel_size=self.kernel_size, avg=avg, std=std)
        self.conv7_bias = create_conv_bias(pn_size*pn_size)
        
        
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn4 = nn.BatchNorm2d(2*channels)
        self.bn7 = nn.BatchNorm2d(pn_size*pn_size)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, input_src, input_thin):
        input_all = torch.cat((input_src, input_thin), dim=1)  # [b, 2, h, w] ##
        try:
            assert input_all.size()[1] == 2  # ##
        except AssertionError:
            raise ValueError(f'Expect concatenated confidence map to have channel depth of 3. Got {input_all.shape} instead.')
        # Full padding 
        fm_1 = F.conv2d(input_all, self.conv1_kernel, padding=self.kernel_size//2)
        fm_1 = self.bn1(fm_1)
        fm_1 = self.relu(fm_1)
        
        fm_4 = F.conv2d(fm_1, self.conv4_kernel, padding=self.kernel_size//2)
        fm_4 = self.bn4(fm_4)
        fm_4 = self.relu(fm_4)
        
        fm_7 = F.conv2d(fm_4, self.conv7_kernel, self.conv7_bias, padding=self.kernel_size//2)
        fm_7 = self.bn7(fm_7)
        fm_7 = F.relu(fm_7)

        return F.softmax(fm_7, dim=1)  # [b, pn_size * pn_size, h, w]


class adaptive_aggregationC(nn.Module):
    def __init__(self, pn_size=5):
        """
        :param pn_size: the size of propagation neighbors.
        
        Aggregating confidence map with agg.coeff.
        """
        super().__init__()
        self.kernel_size = pn_size
        self.weight = create_mapping_kernel(kernel_size=self.kernel_size)

    def forward(self, input_thin, agg_coeff):
        # agg.coeff is gained from progating 
        assert input_thin.size()[1] == 1
        # Just maximize from both map.
        input_sal = input_thin
        # Mapping...
        map_sal = F.conv2d(input_sal, self.weight, padding=self.kernel_size//2)
        # map_sal_inv = 1.0 - map_sal
        assert agg_coeff.size() == map_sal.size()

        # Calculate each pixels product.
        prod_sal = torch.sum(map_sal * agg_coeff, dim=1).unsqueeze(1)
        # prod_sal = F.sigmoid(prod_sal)
        # prod_sal_inv = torch.sum(map_sal_inv * agg_coeff, dim=1).unsqueeze(1)

        return prod_sal # [b, 1, h, w]

class baseMulti(nn.Module):
    def __init__(self, in_channels=3, channels=256, pn_size=5, kernel_size=3, avg=0.0, std=0.1):
        """
        Base for single branch coarse stage
        :param channels: the basic channels of feature maps.
        :param pn_size: the size of propagation neighbors.
        :param kernel_size: the size of kernel.
        :param avg: the mean of normal initialization.
        :param std: the standard deviation of normal initialization.
        
        Documentation:
        this is a propagation basis class.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels

        self.conv1_kernel = create_conv_kernel(in_channels=in_channels, out_channels=channels,
                                               kernel_size=self.kernel_size, avg=avg, std=std)
        # conv1 -> conv4 propagation kernel
        self.conv4_kernel = create_conv_kernel(in_channels=channels, out_channels=2*channels,
                                               kernel_size=self.kernel_size, avg=avg, std=std)
        # conv4 -> conv 7 propagation kernel
        self.conv7_kernel = create_conv_kernel(in_channels=2*channels, out_channels=pn_size*pn_size,
                                               kernel_size=self.kernel_size, avg=avg, std=std)
        self.conv7_bias = create_conv_bias(pn_size*pn_size)
        
        
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn4 = nn.BatchNorm2d(2*channels)
        self.bn7 = nn.BatchNorm2d(pn_size*pn_size)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, basis: Sequence[Tensor]):
        input_all = torch.cat(basis[0:self.in_channels], dim=1)  # [b, 2, h, w] ##
        assert input_all.shape[1] == self.in_channels, f'Expect the input to had {self.in_channels} channels, got {input_all.shape[1]} instead.'
        # Full padding 
        fm_1 = F.conv2d(input_all, self.conv1_kernel, padding=self.kernel_size//2)
        fm_1 = self.bn1(fm_1)
        fm_1 = self.relu(fm_1)
        
        fm_4 = F.conv2d(fm_1, self.conv4_kernel, padding=self.kernel_size//2)
        fm_4 = self.bn4(fm_4)
        fm_4 = self.relu(fm_4)
        
        fm_7 = F.conv2d(fm_4, self.conv7_kernel, self.conv7_bias, padding=self.kernel_size//2)
        fm_7 = self.bn7(fm_7)
        fm_7 = F.relu(fm_7)

        return F.softmax(fm_7, dim=1)  # [b, pn_size * pn_size, h, w]


class adaptive_aggregationMulti(nn.Module):
    def __init__(self, pn_size=5):
        """
        :param pn_size: the size of propagation neighbors.
        
        Aggregating confidence map with agg.coeff.
        """
        super().__init__()
        self.kernel_size = pn_size
        self.weight = create_mapping_kernel(kernel_size=self.kernel_size)

    def forward(self, max_prob_input, agg_coeff):
        # agg.coeff is gained from progating 
        assert max_prob_input.shape[1] == 1
        input_sal = max_prob_input
        # Mapping...
        map_sal = F.conv2d(input_sal, self.weight, padding=self.kernel_size//2)
        # map_sal_inv = 1.0 - map_sal
        assert agg_coeff.size() == map_sal.size()

        # Calculate each pixels product.
        prod_sal = torch.sum(map_sal * agg_coeff, dim=1).unsqueeze(1)
        # prod_sal = F.sigmoid(prod_sal)
        # prod_sal_inv = torch.sum(map_sal_inv * agg_coeff, dim=1).unsqueeze(1)

        return prod_sal # [b, 1, h, w]


class GlobalAveragePooling2D(nn.Module):

    def __init__(self):
        super().__init__()
    def forward(self, x: Tensor):
        return x.mean(dim=(2, 3))