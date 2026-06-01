import torch.nn as nn

from models.vision_transformer import SwinUnet

from types import SimpleNamespace

def build_config():
    cfg = SimpleNamespace()

    cfg.DATA = SimpleNamespace()
    cfg.DATA.IMG_SIZE = 256

    cfg.TRAIN = SimpleNamespace()
    cfg.TRAIN.USE_CHECKPOINT = False

    cfg.MODEL = SimpleNamespace()
    cfg.MODEL.DROP_RATE = 0.0
    cfg.MODEL.DROP_PATH_RATE = 0.2

    cfg.MODEL.SWIN = SimpleNamespace()
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.IN_CHANS = 3
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2,2,2,2]
    cfg.MODEL.SWIN.DECODER_DEPTHS = [2,2,2,1]
    cfg.MODEL.SWIN.NUM_HEADS = [3,6,12,24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 8

    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True

    return cfg

class SwinUNetTiny(nn.Module):

    def __init__(self):
        super().__init__()

        cfg = build_config()

        self.model = SwinUnet(
            config=cfg,
            img_size=256,
            num_classes=1
        )

    def forward(self, x):
        return self.model(x)