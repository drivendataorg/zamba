from timesformer.models.vit import TimeSformer as TimeSformerBase
import torch

from zamba_algorithms.pytorch_lightning.utils import ZambaVideoClassificationLightningModule
from zamba_algorithms.settings import MODEL_DIRECTORY


class TimeSformer(ZambaVideoClassificationLightningModule):
    def __init__(
        self,
        num_classes: int = 24,
        backbone_mode: str = "eval",
        **kwargs,
    ):
        """Pretrained TimeSformer model for fine-tuning with the following architecture:

        Input -> TimeSformer Backbone -> Head -> Output

        Args:
            num_classes (int)
            freeze_backbone (bool): If True, freeze the backbone TimeSformer model so it acts as a
                feature extractor. If False, train the TimeSformer backbone.
        """
        super().__init__(num_classes=num_classes, **kwargs)
        model = TimeSformerBase(
            img_size=224,
            num_classes=num_classes,
            num_frames=8,
            attention_type="divided_space_time",
            pretrained_model=str(MODEL_DIRECTORY / "TimeSformer_divST_8x32_224_K400.pyth"),
        )

        # Simplest to freeze everything then unfreeze the head
        for name, module in model.model.named_modules():
            for param in module.parameters():
                param.requires_grad = False
        for param in model.model.head.parameters():
            param.requires_grad = True

        # self.backbone attribute lets `BackboneFinetune` freeze and unfreeze that module
        self.backbone = torch.nn.ModuleList([model.model.blocks[-1:], model.model.norm])

        self.model = model
        self.backbone_mode = backbone_mode

        self.save_hyperparameters()

    def forward(self, x, *args, **kwargs):
        # Simplest to set everything to eval then set head to train if it was before
        if self.backbone_mode == "eval":
            head_is_training = self.model.model.head.training
            for name, module in self.model.named_modules():
                module.eval()
            if head_is_training:
                self.model.model.head.train()

        return self.model(x)
