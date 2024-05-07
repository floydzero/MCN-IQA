from model.capsnet import *
from model.backbone import resnet50_4stage
from config import cfg


class MixerCaps(nn.Module):
    """
    Main model:{MixerCaps}, load settings from MixerCaps/config/MixerCaps.yaml
    """
    def __init__(self):
        super().__init__()
        self.resnet = resnet50_4stage(pretrained=True)

        self.content = ContentCaps(feature_in_c=cfg.CAPSNET.CONTENT.FEATURE_IN_C,
                                   spatial_in_c=cfg.CAPSNET.CONTENT.SPATIAL_IN_C,
                                   spatial_in_dim=cfg.CAPSNET.CONTENT.SPATIAL_IN_DIM,
                                   feature_in_dim=cfg.CAPSNET.CONTENT.FEATURE_IN_DIM,
                                   spatial_out_dim=cfg.CAPSNET.CONTENT.SPATIAL_OUT_DIM,
                                   feature_out_dim=cfg.CAPSNET.CONTENT.FEATURE_OUT_DIM,
                                   num=cfg.CAPSNET.CONTENT.NUM,
                                   learnable=cfg.CAPSNET.CONTENT.LEARNABLE)

        self.content.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.resnet(x)
        pred = self.content([out[2], out[3]])
        #pred,dis = self.content([out[2],out[3]])
        # if cfg.CAPSNET.IF_CLASSIFICATION:
        #     return pred, dis
        return pred
        #return pred,dis

