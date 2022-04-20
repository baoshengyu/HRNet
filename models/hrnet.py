
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


__all__ = [ 'hrnet18', 'hrnet32', 'hrnet48' ]


model_urls = {
    'hrnet18': 'hrnet18-699e7ab89.pth',
    'hrnet32': 'hrnet32-9f864d2d6.pth',
    'hrnet48': '',
}


class BasicBlock(nn.Module):
    """BasicBlock
    """
    expansion = 1
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out))+residual)
        return out


class Bottleneck(nn.Module):
    """Bottleneck
    """
    expansion = 4
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, self.expansion*out_channels, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*out_channels)
        self.downsample = None
        if in_channels != out_channels*self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*self.expansion, 1, 1, 0),
                nn.BatchNorm2d(out_channels*self.expansion),
            )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(residual)
        out = F.relu(out+residual)
        return out


class MRStreamBlock(nn.Module):
    """Multi Resolution Stream Block
    """
    def __init__(self, block, in_channels, out_channels, num_streams, num_blocks):
        super(MRStreamBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stream_list = self._make_layers(block, num_streams, num_blocks)

    def _make_layers(self, block, num_streams, num_blocks):
        stream_list = nn.ModuleList()
        for i in range(num_streams):
            stream = [block(self.in_channels*(2**i), self.out_channels*(2**i))]
            in_channels = self.out_channels*block.expansion*(2**i)
            for j in range(num_blocks-1):
                stream.append(block(in_channels, self.out_channels*(2**i)))
            stream_list.append(nn.Sequential(*stream))
        return stream_list

    def forward(self, x_list):
        return [f(x) for f,x in zip(self.stream_list, x_list)]


class MRFusionBlock(nn.Module):
    """Multi Resolution Fusion Block
    """
    def __init__(self, in_channels=18, out_channels=18, num_streams=2):
        super(MRFusionBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fusion_list = self._make_layers(num_streams)

    def _make_layers(self, num_streams):
        fusion_list = nn.ModuleList()
        for i in range(num_streams):
            out_channels = self.out_channels*(2**i)
            layer_list = nn.ModuleList()
            for j in range(num_streams):
                in_channels = self.in_channels*(2**j)
                if j > i:
                    layer_list.append(nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                        nn.BatchNorm2d(out_channels),
                        nn.Upsample(scale_factor=2**(j-i), mode='bilinear', align_corners=False),
                    ))
                elif j == i:
                    layer_list.append(nn.Identity())
                else:
                    layer = []
                    for k in range(i-j):
                        layer.append(nn.Sequential(
                            nn.Conv2d(in_channels, out_channels if k==i-j-1 else in_channels, 3, 2, 1),
                            nn.BatchNorm2d(out_channels if k==i-j-1 else in_channels),
                            nn.Identity() if k==(i-j-1) else nn.ReLU(inplace=True),
                        ))
                    layer_list.append(nn.Sequential(*layer))
            fusion_list.append(layer_list)
        return fusion_list

    def forward(self, x_list):
        y_list = []
        for layer_list in self.fusion_list:
            out = [f(x) for f,x in zip(layer_list, x_list)]
            out = F.relu(torch.stack(out).sum(0))
            y_list.append(out)
        return y_list


class MRConvModule(nn.Module):
    """High Resolution Convolution Module
    """
    def __init__(self, block, in_channels=64, out_channels=64, num_streams=2, num_blocks=4):
        super(MRConvModule, self).__init__()
        self.stream_layer = MRStreamBlock(block, in_channels, out_channels, num_streams, num_blocks)
        self.fusion_layer = MRFusionBlock(in_channels, out_channels, num_streams)

    def forward(self, x_list):
        y_list = self.stream_layer(x_list)
        y_list = self.fusion_layer(y_list)
        return y_list


class MRPoolModule(nn.Module):
    """Muti Resolution Transform Module
    """
    def __init__(self, in_channels, out_channels, num_streams):
        super(MRPoolModule, self).__init__()
        self.transform_layers = nn.ModuleList()
        for i in range(num_streams):
            if in_channels != out_channels:
                self.transform_layers.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels*(2**i), 3, 1, 1, bias=False),
                    nn.BatchNorm2d(out_channels*(2**i)),
                    nn.ReLU(inplace=True),
                ))
            else:
                self.transform_layers.append(nn.Identity())
        self.transform_layers.append(nn.Sequential(
            nn.Conv2d(in_channels*(2**(num_streams-1)), out_channels*(2**num_streams), 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels*(2**num_streams)),
            nn.ReLU(inplace=True),
        ))

    def forward(self, x_list):
        y_list = [f(x) for f,x in zip(self.transform_layers, x_list)]
        y_new = self.transform_layers[-1](x_list[-1])
        y_list.append(y_new)
        return y_list


class CLSHead(nn.Module):
    """CLSHead
    """
    def __init__(self, num_classes=1000, in_channels=18, out_channels=32, num_streams=4):
        super(CLSHead, self).__init__()
        self.smooth_layers = nn.ModuleList([
            Bottleneck(in_channels*(2**i), out_channels*(2**i)) for i in range(num_streams)
        ])
        out_channels *= Bottleneck.expansion
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels*(2**i), out_channels*(2**(i+1)), 3, 2, 1, bias=False),
                nn.BatchNorm2d(out_channels*(2**(i+1))),
                nn.ReLU(inplace=True),
            )
            for i in range(num_streams-1)
        ])
        self.final_layer = nn.Sequential(
            nn.Conv2d(out_channels*8, 2048, 1, 1, 0, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2048, 1000, 1, 1, 0),
        )

    def forward(self, x_list):
        y = self.smooth_layers[0](x_list[0])
        for i in range(len(self.smooth_layers)-1):
            y = self.smooth_layers[i+1](x_list[i+1]) + self.downsample_layers[i](y)
        y = self.final_layer(y)
        return y.view(len(y),-1)



class HRNet(nn.Module):
    """HighResolutionNet.
    """
    def __init__(self, 
            blocks=[Bottleneck, BasicBlock, BasicBlock, BasicBlock],
            num_channels=[64,18,18,18], 
            num_streams=[1,2,3,4], 
            num_modules=[1,1,4,3],
            num_blocks=[4,4,4,4],
        ):
        super(HRNet, self).__init__()
        self.num_channels = num_channels
        self.num_streams = num_streams
        self.num_modules = num_modules
        self.num_blocks = num_blocks
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv1 = self._make_conv_module(blocks[0], 0)
        self.pool1 = self._make_pool_module(blocks[0], 0)

        self.conv2 = self._make_conv_module(blocks[1], 1)
        self.pool2 = self._make_pool_module(blocks[1], 1)

        self.conv3 = self._make_conv_module(blocks[2], 2)
        self.pool3 = self._make_pool_module(blocks[2], 2)

        self.conv4 = self._make_conv_module(blocks[3], 3)

        self.classifier = CLSHead(num_classes=1000, in_channels=num_channels[-1])

    def _make_conv_module(self, block, stage):
        conv_modules = []
        for i in range(self.num_modules[stage]):
            conv_modules.append(MRConvModule(block, 
                self.num_channels[stage], self.num_channels[stage], 
                self.num_streams[stage], self.num_blocks[stage]))
        return nn.Sequential(*conv_modules)

    def _make_pool_module(self, block, stage):
        return MRPoolModule(self.num_channels[stage]*block.expansion, 
            self.num_channels[stage+1], self.num_streams[stage])

    def forward(self, x):
        x_list = [self.conv0(x)]
        x_list = self.pool1(self.conv1(x_list))
        x_list = self.pool2(self.conv2(x_list))
        x_list = self.conv3(x_list)
        x_list = self.pool3(x_list)
        x_list = self.conv4(x_list)
        x_list = self.classifier(x_list)
        return x_list 


###############################
def hrnet18(pretrained=False, **kwargs):
    model = HRNet(
        blocks = [Bottleneck, BasicBlock, BasicBlock, BasicBlock],
        num_modules = [1, 1, 4, 3],
        num_streams = [1, 2, 3, 4],
        num_blocks = [4, 4, 4, 4],
        num_channels = [64, 18, 18, 18],
        **kwargs
    )
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['hrnet18']), strict=False)
    return model


def hrnet32(pretrained=False, **kwargs):
    model = HRNet(
        blocks = [Bottleneck, BasicBlock, BasicBlock, BasicBlock],
        num_modules = [1, 1, 4, 3],
        num_streams = [1, 2, 3, 4],
        num_blocks = [4, 4, 4, 4],
        num_channels = [64, 32, 32, 32],
        **kwargs
    )
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['hrnet32']), strict=False)
    return model


def hrnet48(pretrained=False, **kwargs):
    model = HRNet(
        blocks = [Bottleneck, BasicBlock, BasicBlock, BasicBlock],
        num_modules = [1, 1, 4, 3],
        num_streams = [1, 2, 3, 4],
        num_blocks = [4, 4, 4, 4],
        num_channels = [64, 48, 48, 48],
        **kwargs
    )
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['hrnet48']), strict=False)
    return model



if __name__ == '__main__':

    model = hrnet48()
    print(model)
    from thop import profile, clever_format
    macs, params = profile(model, inputs=(torch.randn(1, 3, 224, 224),))
    macs, params = clever_format([macs, params], "%.2f")
    print(macs, params)


