import pytorch_lightning as pl

from pytorch_lightning.callbacks import RichProgressBar

class LightningProgressBar(RichProgressBar):

    def get_metrics(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # don't show the version number
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items