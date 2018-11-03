""" Module containing custom layers """
import torch as th
import copy


# ==========================================================
# Additional layers defining the operations performed
# by the Generator and the Discriminator
# ==========================================================

class AddCoords(th.nn.Module):
    """
        Module for concatenating coordinate channels to the
        input network volume
        args:
            :param with_r: whether to use radial coordinates
                           (boolean) [default = True]
    """

    def __init__(self, with_r=True):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        forward pass of the Module
        :param input_tensor: input tensor [b x c x h x w]
        :return: out => output tensor [b x (c + 2 / 3) x h x w]
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = th.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = th.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        if x_dim != 1 and y_dim != 1:
            xx_channel = xx_channel.float() / (x_dim - 1)
            yy_channel = yy_channel.float() / (y_dim - 1)

            xx_channel = xx_channel * 2 - 1
            yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        out = th.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = th.sqrt(th.pow(xx_channel.type_as(input_tensor) - 0.5, 2)
                         + th.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            out = th.cat([out, rr], dim=1)

        return out


class CoordConv(th.nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, with_r=True, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = th.nn.Conv2d(in_size, out_channels,
                                 kernel_size, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


class CoordConvTranspose(th.nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, with_r=True, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = th.nn.ConvTranspose2d(in_size, out_channels,
                                          kernel_size, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


# ==========================================================
# Layers required for Building The generator and
# discriminator
# ==========================================================

class GenInitialBlock(th.nn.Module):
    """ Module implementing the initial block of the Generator
        Takes in whatever latent size and generates output volume
        of size 4 x 4
    """

    def __init__(self, in_channels, use_coord_conv=True):
        """
        constructor for the inner class
        :param in_channels: number of input channels to the block
        :param use_coord_conv: whether to use coord_conv or not
        """
        from torch.nn import LeakyReLU
        if use_coord_conv:
            Conv, ConvT = CoordConv, CoordConvTranspose
        else:
            from torch.nn import Conv2d as Conv, ConvTranspose2d as ConvT
        super().__init__()

        self.conv_1 = ConvT(in_channels, in_channels, (4, 4), bias=True)
        self.conv_2 = Conv(in_channels, in_channels, (3, 3), padding=(1, 1), bias=True)

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the block
        :param x: input to the module
        :return: y => output
        """
        # convert the tensor shape:
        y = x.view(*x.shape, 1, 1)  # add two dummy dimensions for
        # convolution operation

        # perform the forward computations:
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))

        return y


class GenGeneralConvBlock(th.nn.Module):
    """ Module implementing a general convolutional block """

    def __init__(self, in_channels, out_channels, use_coord_conv=True,
                 use_upsampling=False):
        """
        constructor for the class
        :param in_channels: number of input channels to the block
        :param out_channels: number of output channels required
        :param use_coord_conv: whether to use coord_conv
        :param use_upsampling: whether to use upsampling or to use
                               Transpose Conv.
        """
        from torch.nn import LeakyReLU

        super().__init__()

        if use_coord_conv:
            Conv, ConvT = CoordConv, CoordConvTranspose
        else:
            from torch.nn import Conv2d as Conv, ConvTranspose2d as ConvT
        self.conv_1 = Conv(in_channels, out_channels, (3, 3),
                           padding=1, bias=True)
        self.conv_2 = Conv(out_channels, out_channels, (3, 3),
                           padding=1, bias=True)

        if use_upsampling:
            self.upsampler = ConvT(in_channels, in_channels, (4, 4), stride=2)

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the block
        :param x: input
        :return: y => output
        """
        from torch.nn.functional import interpolate

        if hasattr(self, "upsampler"):
            y = self.upsampler(x)
        else:
            y = interpolate(x, scale_factor=2)
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))

        return y


class MinibatchStdDev(th.nn.Module):
    def __init__(self, averaging='all'):
        """
        constructor for the class
        :param averaging: the averaging mode used for calculating the MinibatchStdDev
        """
        super().__init__()

        # lower case the passed parameter
        self.averaging = averaging.lower()

        if 'group' in self.averaging:
            self.n = int(self.averaging[5:])
        else:
            assert self.averaging in [
                'all', 'flat', 'spatial', 'none', 'gpool'], \
                'Invalid averaging mode %s' % self.averaging

        # calculate the std_dev in such a way that it doesn't result in 0
        # otherwise 0 norm operation's gradient is nan
        self.adjusted_std = lambda x, **kwargs: th.sqrt(
            th.mean((x - th.mean(x, **kwargs)) ** 2, **kwargs) + 1e-8)

    def forward(self, x):
        """
        forward pass of the Layer
        :param x: input
        :return: y => output
        """
        shape = list(x.size())
        target_shape = copy.deepcopy(shape)

        # compute the std's over the minibatch
        vals = self.adjusted_std(x, dim=0, keepdim=True)

        # perform averaging
        if self.averaging == 'all':
            target_shape[1] = 1
            vals = th.mean(vals, dim=1, keepdim=True)

        elif self.averaging == 'spatial':
            if len(shape) == 4:
                vals = th.mean(th.mean(vals, 2, keepdim=True), 3, keepdim=True)

        elif self.averaging == 'none':
            target_shape = [target_shape[0]] + [s for s in target_shape[1:]]

        elif self.averaging == 'gpool':
            if len(shape) == 4:
                vals = th.mean(th.mean(th.mean(x, 2, keepdim=True),
                                       3, keepdim=True), 0, keepdim=True)
        elif self.averaging == 'flat':
            target_shape[1] = 1
            vals = th.FloatTensor([self.adjusted_std(x)])

        else:  # self.averaging == 'group'
            target_shape[1] = self.n
            vals = vals.view(self.n, self.shape[1] /
                             self.n, self.shape[2], self.shape[3])
            vals = th.mean(vals, 0, keepdim=True).view(1, self.n, 1, 1)

        # spatial replication of the computed statistic
        vals = vals.expand(*target_shape)

        # concatenate the constant feature map to the input
        y = th.cat([x, vals], 1)

        # return the computed value
        return y


class DisFinalBlock(th.nn.Module):
    """ Final block for the Discriminator """

    def __init__(self, in_channels, use_coord_conv=True):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param use_coord_conv: whether to use coord_conv [default = True]
        """
        from torch.nn import LeakyReLU
        if use_coord_conv:
            Conv = CoordConv
        else:
            from torch.nn import Conv2d as Conv

        super().__init__()

        # declare the required modules for forward pass
        self.batch_discriminator = MinibatchStdDev()

        # modules required:
        self.conv_1 = Conv(in_channels + 1, in_channels, (3, 3),
                           padding=1, bias=True)
        self.conv_2 = Conv(in_channels, in_channels, (4, 4), bias=True)

        # final conv layer emulates a fully connected layer
        self.conv_3 = Conv(in_channels, 1, (1, 1), bias=True)

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the FinalBlock
        :param x: input
        :return: y => output
        """
        # minibatch_std_dev layer
        y = self.batch_discriminator(x)

        # define the computations
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))

        # fully connected layer
        y = self.conv_3(y)  # This layer has linear activation

        # flatten the output raw discriminator scores
        return y.view(-1)


class DisGeneralConvBlock(th.nn.Module):
    """ General block in the discriminator  """

    def __init__(self, in_channels, out_channels, use_coord_conv=True):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param use_coord_conv: whether to use coord_conv [default=True]
        """
        from torch.nn import AvgPool2d, LeakyReLU

        if use_coord_conv:
            Conv = CoordConv
        else:
            from torch.nn import Conv2d as Conv

        super().__init__()

        # convolutional modules
        self.conv_1 = Conv(in_channels, in_channels, (3, 3),
                           padding=1, bias=True)
        self.conv_2 = Conv(in_channels, out_channels, (3, 3),
                           padding=1, bias=True)
        self.downSampler = AvgPool2d(2)  # downsampler

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the module
        :param x: input
        :return: y => output
        """
        # define the computations
        y = self.lrelu(self.conv_1(x))
        y = self.lrelu(self.conv_2(y))
        y = self.downSampler(y)

        return y
