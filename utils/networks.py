import torch
import torch.nn as nn
import torchvision
from pathlib import Path
from utils import experiment_manager
from copy import deepcopy
from collections import OrderedDict
from sys import stderr


def save_checkpoint(network, optimizer, epoch, cfg: experiment_manager.CfgNode):
    save_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}.pt'
    checkpoint = {
        'epoch': epoch,
        'network': network.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, save_file)


def load_checkpoint(cfg: experiment_manager.CfgNode, device: torch.device):
    net = PopulationDualTaskNet(cfg.MODEL) if cfg.CHANGE_DETECTION.ENDTOEND else PopulationNet(cfg.MODEL)
    net.to(device)

    save_file = Path(cfg.PATHS.OUTPUT) / 'networks' / f'{cfg.NAME}.pt'
    checkpoint = torch.load(save_file, map_location=device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.TRAINER.LR, weight_decay=0.01)

    net.load_state_dict(checkpoint['network'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return net, optimizer, checkpoint['epoch']


def load_weights(output_path: Path, config_name: str, device: torch.device):
    save_file = Path(output_path) / 'networks' / f'{config_name}.pt'
    checkpoint = torch.load(save_file, map_location=device)
    return checkpoint['network']


class PopulationDualTaskNet(nn.Module):

    def __init__(self, model_cfg: experiment_manager.CfgNode):
        super(PopulationDualTaskNet, self).__init__()

        self.encoder = PopulationNet(model_cfg)
        self.encoder.enable_fc = False
        n_features = self.encoder.model.fc.in_features
        self.change_fc = nn.Linear(n_features, 1)
        self.relu = torch.nn.ReLU()

        self.dummy_input_t1 = torch.rand((2, 4, 10, 10)).to('cuda')
        self.dummy_input_t2 = torch.rand((2, 4, 10, 10)).to('cuda')

    def forward(self, x_t1: torch.Tensor, x_t2: torch.Tensor) -> tuple:
        features_t1 = self.encoder(x_t1)
        features_t2 = self.encoder(x_t2)
        p_t1 = self.relu(self.encoder.model.fc(features_t1))
        p_t2 = self.relu(self.encoder.model.fc(features_t2))
        features_fusion = features_t2 - features_t1
        p_change = self.change_fc(features_fusion)
        return p_change, p_t1, p_t2

    def load_pretrained_encoder(self, cfg_name: str, weights_path: Path, device: torch.device, verbose: bool = True):
        pretrained_weights = load_weights(weights_path, cfg_name, device)
        self.encoder.load_state_dict(pretrained_weights)
        if verbose:
            print(f'Loaded encoder weights from {cfg_name}!')

    def freeze_encoder(self, freeze_fc: bool = True, freeze_bn_rmean: bool = True):
        # https://discuss.pytorch.org/t/network-output-changes-even-when-freezed-during-training/36423/6
        # https://discuss.pytorch.org/t/how-to-freeze-bn-layers-while-training-the-rest-of-network-mean-and-var-wont-freeze/89736/17
        for m in self.encoder.modules():
            for param in m.parameters():
                param.requires_grad = False
            if freeze_bn_rmean:
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.track_running_stats = False
                    # m.eval()
        if freeze_fc:
            self.encoder.model.fc.requires_grad = False

    def print_weight_stamps(self):
        enc_stamp = 0
        for parameter in self.encoder.parameters():
            enc_stamp += torch.sum(parameter).item()

        net_stamp = 0
        for parameter in self.parameters():
            net_stamp += torch.sum(parameter).item()
        print(f'Weight stamps: net {net_stamp:.3f}, enc {enc_stamp:.3f}')

    def print_output_stamps(self):
        training = self.training
        if self.training:
            self.eval()
        with torch.no_grad():
            features_t1 = self.encoder(self.dummy_input_t1)
            features_t2 = self.encoder(self.dummy_input_t2)

            p_t1 = self.relu(self.encoder.model.fc(features_t1))
            p_t2 = self.relu(self.encoder.model.fc(features_t2))

            features_fusion = features_t2 - features_t1
            p_change = self.change_fc(features_fusion)

            # out_diff, b, c = self.forward(self.dummy_input_t1, self.dummy_input_t2)

        print(f'Enc output stamp {torch.sum(p_t1).item():.3f}, {torch.sum(p_t2).item():.3f}')
        print(f'Net output stamp {torch.sum(p_change).item():.3f}')

        if training:
            self.train()


class PopulationNet(nn.Module):

    def __init__(self, model_cfg, enable_fc: bool = True):
        super(PopulationNet, self).__init__()
        self.model_cfg = model_cfg
        self.enable_fc = enable_fc
        pt = model_cfg.PRETRAINED
        assert (model_cfg.TYPE == 'resnet')
        if model_cfg.SIZE == 18:
            self.model = torchvision.models.resnet18(pretrained=pt)
        elif model_cfg.SIZE == 50:
            self.model = torchvision.models.resnet50(pretrained=pt)
        else:
            raise Exception(f'Unkown resnet size ({model_cfg.SIZE}).')

        new_in_channels = model_cfg.IN_CHANNELS

        if new_in_channels != 3:
            # only implemented for resnet
            assert (model_cfg.TYPE == 'resnet')

            first_layer = self.model.conv1
            # Creating new Conv2d layer
            new_first_layer = nn.Conv2d(
                in_channels=new_in_channels,
                out_channels=first_layer.out_channels,
                kernel_size=first_layer.kernel_size,
                stride=first_layer.stride,
                padding=first_layer.padding,
                bias=first_layer.bias
            )
            # he initialization
            nn.init.kaiming_uniform_(new_first_layer.weight.data, mode='fan_in', nonlinearity='relu')
            if new_in_channels > 3:
                # replace weights of first 3 channels with resnet rgb ones
                first_layer_weights = first_layer.weight.data.clone()
                new_first_layer.weight.data[:, :first_layer.in_channels, :, :] = first_layer_weights
            # if it is less than 3 channels we use he initialization (no pretraining)

            # replacing first layer
            self.model.conv1 = new_first_layer
            # https://discuss.pytorch.org/t/how-to-change-no-of-input-channels-to-a-pretrained-model/19379/2
            # https://discuss.pytorch.org/t/how-to-modify-the-input-channels-of-a-resnet-model/2623/10

        # replacing fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, model_cfg.OUT_CHANNELS)
        self.relu = torch.nn.ReLU()
        self.encoder = torch.nn.Sequential(*(list(self.model.children())[:-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_fc:
            x = self.model(x)
            x = self.relu(x)
        else:
            x = self.encoder(x)
            x = x.squeeze()
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
        return x


if __name__ == '__main__':
    x = torch.randn(1, 5, 224, 224)
    model = torchvision.models.vgg16(pretrained=False)  # pretrained=False just for debug reasons
    first_conv_layer = [nn.Conv2d(5, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
    first_conv_layer.extend(list(model.features))
    model.features = nn.Sequential(*first_conv_layer)
    output = model(x)