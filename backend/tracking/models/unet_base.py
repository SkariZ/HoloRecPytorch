import deeplay as dl
from torch.nn import LeakyReLU, InstanceNorm2d

def UNetBase(in_channels: int, out_channels: int):
    model = dl.UNet2d(
        in_channels=in_channels,
        channels=[16, 32, 64],
        out_channels=out_channels,
    )

    model["encoder", ..., "activation"].configure(
        LeakyReLU, negative_slope=0.2
    )
    model["decoder", ..., "activation#:-1"].configure(
        LeakyReLU, negative_slope=0.2
    )
    model["decoder", "blocks", :-1].all.normalized(
        InstanceNorm2d
    )
    model[..., "blocks"].configure(
        order=["layer", "normalization", "activation"]
    )

    model.build()

    return model