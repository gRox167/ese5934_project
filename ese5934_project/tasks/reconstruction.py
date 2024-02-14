from typing import Any

import lightning.pytorch as pl
import torch
from einops import rearrange


class Recon(pl.LightningModule):
    def __init__(
        self,
        field,
        lr=1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # self.automatic_optimization = False
        self.field = field
        self.lr = lr

    def training_step(self, batch, batch_idx):
        recon_opt = self.optimizers()
        kspace_data_odd, kspace_traj_odd = (
            batch["kspace_data_z_fixed"],
            batch["kspace_traj_fixed"],
        )
        kspace_data_even, kspace_traj_even = (
            batch["kspace_data_z_moved"],
            batch["kspace_traj_moved"],
        )

        ref_idx = torch.randint(0, 5, (1,))

        loss_a2b, params, image_list = self.n2n_step(
            kspace_data_odd,
            kspace_traj_odd,
            kspace_data_even,
            kspace_traj_even,
            ref_idx,
        )
        recon_opt.zero_grad()

        # calculate the loss from even phase to odd phase
        loss_b2a, params, image_list = self.n2n_step(
            kspace_data_even,
            kspace_traj_even,
            kspace_data_odd,
            kspace_traj_odd,
            ref_idx,
        )
        self.manual_backward(loss_b2a)
        self.log_dict({"recon/recon_loss": loss_b2a})
        recon_opt.step()

        image, csm, mvf = params["image"], params["csm"], params["mvf"]
        # if self.global_step % 1 == 0:
        for ch in [0, 3, 5]:
            to_png(
                self.trainer.default_root_dir + f"/csm_ch{ch}.png",
                self.recon_module.forward_model.S._csm[0, 0, ch, 0, :, :],
            )  # , vmin=0, vmax=2)
        for i in range(image.shape[0]):
            # to_png(self.trainer.default_root_dir+f'/image_recon{i}.png',
            #        image[i, 0, 0, :, :], vmin=0, vmax=5)
            for j, img in enumerate(image_list):
                to_png(
                    self.trainer.default_root_dir + f"/image_iter_{i}_{j}.png",
                    img[i, 0, 0, :, :],
                    vmin=0,
                    vmax=5,
                )
                # to_png(self.trainer.default_root_dir+f'/image_recon_ph{i}.png',
                #        image_recon_moved[i, 0, :, :])  # , vmin=0, vmax=2)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        ...

    def predict_step(
        self,
        batch: Any,
        batch_idx: int = 0,
        dataloader_idx: int = 0,
        device=torch.device("cuda"),
    ) -> Any:
        ...
        
    def configure_optimizers(self):
        recon_optimizer = torch.optim.AdamW(
            self.parameters(),
            # [
            #     {"params": self.recon_module.parameters()},
            #     # {"params": self.cse_module.parameters()},
            # ],
            lr=self.lr,
        )
        return recon_optimizer
