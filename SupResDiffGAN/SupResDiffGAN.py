import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import time


class SupResDiffGAN(pl.LightningModule):
    """SupResDiffGAN class for Super-Resolution Diffusion Generative Adversarial Network.

    Parameters
    ----------
    ae : nn.Module
        Autoencoder model.
    discriminator : nn.Module
        Discriminator model.
    unet : nn.Module
        UNet generator model.
    diffusion : nn.Module
        Diffusion model.
    learning_rate : float, optional
        Learning rate for the optimizers (default is 1e-4).
    alfa_perceptual : float, optional
        Weight for the perceptual loss (default is 1e-3).
    alfa_adv : float, optional
        Weight for the adversarial loss (default is 1e-2).
    vgg_loss : nn.Module | None, optional
        The VGG loss module for perceptual loss (default is None).
        If None, the perceptual loss will not be used.
    """

    def __init__(
        self,
        ae: nn.Module,
        discriminator: nn.Module,
        unet: nn.Module,
        diffusion: nn.Module,
        learning_rate: float = 1e-4,
        alfa_perceptual: float = 1e-3,
        alfa_adv: float = 1e-2,
        vgg_loss: nn.Module | None = None,
    ) -> None:
        super(SupResDiffGAN, self).__init__()
        self.ae = ae
        self.discriminator = discriminator
        self.generator = unet
        self.diffusion = diffusion

        self.vgg_loss = vgg_loss
        self.content_loss = nn.MSELoss()
        self.adversarial_loss = nn.BCEWithLogitsLoss()  # Updated for mixed precision
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(
            self.device
        )

        self.lr = learning_rate
        self.alfa_adv = alfa_adv
        self.alfa_perceptual = alfa_perceptual
        self.betas = (0.9, 0.999)

        for param in self.ae.parameters():
            param.requires_grad = False

        self.automatic_optimization = False  # Disable automatic optimization

        self.test_step_outputs = []
        self.ema_weight = 0.97
        self.ema_mean = 0.5
        self.s = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SupResDiffGAN model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor with shape (batch_size, channels, height, width).
        """
        with torch.no_grad():
            x_lat = (
                self.ae.encode(x).latent_dist.mode().detach()
                * self.ae.config.scaling_factor
            )
        x = self.diffusion.sample(self.generator, x_lat, x_lat.shape)
        with torch.no_grad():
            x_out = self.ae.decode(x / self.ae.config.scaling_factor).sample
        x_out = torch.clamp(x_out, -1, 1)
        return x_out

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Training step for the SupResDiffGAN model.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            Batch of data containing low-resolution and high-resolution images.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        dict
            Dictionary containing the generator and discriminator losses.
        """
        lr_img, hr_img = batch["lr"], batch["hr"]
        optimizer_g, optimizer_d = self.optimizers()

        with torch.no_grad():
            lr_lat = (
                self.ae.encode(lr_img).latent_dist.mode().detach()
                * self.ae.config.scaling_factor
            )
            x0_lat = (
                self.ae.encode(hr_img).latent_dist.mode().detach()
                * self.ae.config.scaling_factor
            )

        timesteps = torch.randint(
            0,
            self.diffusion.timesteps,
            (x0_lat.shape[0],),
            device=x0_lat.device,
            dtype=torch.long,
        )
        x_t = self.diffusion.forward(x0_lat, timesteps)
        alfa_bars = self.diffusion.alpha_bars_torch.to(timesteps.device)[
            timesteps]
        x_gen_0 = self.generator(lr_lat, x_t, alfa_bars)

        s_tensor = torch.tensor(self.s, device=x0_lat.device, dtype=torch.long).expand(
            x0_lat.shape[0]
        )
        x_s = self.diffusion.forward(x0_lat, s_tensor)
        x_gen_s = self.diffusion.forward(x_gen_0, s_tensor)

        with torch.no_grad():
            sr_img = torch.clamp(
                self.ae.decode(
                    x_gen_0 / self.ae.config.scaling_factor).sample, -1, 1
            )
            hr_s_img = torch.clamp(
                self.ae.decode(
                    x_s / self.ae.config.scaling_factor).sample, -1, 1
            )
            sr_s_img = torch.clamp(
                self.ae.decode(
                    x_gen_s / self.ae.config.scaling_factor).sample, -1, 1
            )

        if batch_idx % 2 == 0:
            self.toggle_optimizer(optimizer_g)
            optimizer_g.zero_grad()
            g_loss = self.generator_loss(
                x0_lat, x_gen_0, hr_img, sr_img, hr_s_img, sr_s_img
            )
            self.manual_backward(g_loss)
            optimizer_g.step()
            self.untoggle_optimizer(optimizer_g)
            self.log(
                "train/g_loss",
                g_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            return {"g_loss": g_loss}
        else:
            self.toggle_optimizer(optimizer_d)
            optimizer_d.zero_grad()
            d_loss = self.discriminator_loss(hr_s_img, sr_s_img.detach())
            self.manual_backward(d_loss)
            optimizer_d.step()
            self.untoggle_optimizer(optimizer_d)
            self.log(
                "train/d_loss",
                d_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            return {"d_loss": d_loss}

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Validation step for the SupResDiffGAN model.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            Batch of data containing low-resolution and high-resolution images.
        batch_idx : int
            Index of the batch.
        """
        lr_img, hr_img = batch["lr"], batch["hr"]
        lr_img, hr_img = lr_img.to(self.device), hr_img.to(self.device)
        sr_img = self(lr_img)
        padding_info = {"lr": batch["padding_data_lr"],
                        "hr": batch["padding_data_hr"]}

        # Enhanced visualization for the first 3 batches of every validation epoch
        if batch_idx < 3:
            print(
                f"Generating validation images for Epoch {self.current_epoch}, Batch {batch_idx}..."
            )
            try:
                title = f"Validation Epoch {self.current_epoch} - Batch {batch_idx}"
                per_image_metrics = []
                # Limit to 5 samples max
                for i in range(min(5, lr_img.shape[0])):
                    hr_img_np = hr_img[i].detach(
                    ).cpu().numpy().transpose(1, 2, 0)
                    sr_img_np = sr_img[i].detach(
                    ).cpu().numpy().transpose(1, 2, 0)
                    hr_img_np = (hr_img_np + 1) / 2
                    sr_img_np = (sr_img_np + 1) / 2
                    hr_img_np = hr_img_np[
                        : padding_info["hr"][i][1], : padding_info["hr"][i][0], :
                    ]
                    sr_img_np = sr_img_np[
                        : padding_info["hr"][i][1], : padding_info["hr"][i][0], :
                    ]
                    psnr = peak_signal_noise_ratio(
                        hr_img_np, sr_img_np, data_range=1.0)
                    ssim = structural_similarity(
                        hr_img_np, sr_img_np, channel_axis=-1, data_range=1.0
                    )
                    lpips_val = (
                        self.lpips(hr_img[i: i + 1],
                                   sr_img[i: i + 1]).cpu().item()
                    )
                    per_image_metrics.append((psnr, ssim, lpips_val))

                img_array = self.plot_images_with_metrics(
                    hr_img, lr_img, sr_img, padding_info, title, per_image_metrics
                )
                avg_metrics = [
                    np.mean([m[i] for m in per_image_metrics]) for i in range(3)
                ]
                self.logger.experiment.log(
                    {
                        f"Validation/Epoch_{self.current_epoch}_Batch_{batch_idx}": [
                            wandb.Image(
                                img_array,
                                caption=f"Epoch {self.current_epoch} Batch {batch_idx} - Avg PSNR/SSIM/LPIPS: {avg_metrics[0]:.2f}/{avg_metrics[1]:.3f}/{avg_metrics[2]:.3f}",
                            )
                        ]
                    }
                )
                print(
                    f"Successfully logged validation images for Epoch {self.current_epoch}, Batch {batch_idx}"
                )
            except Exception as e:
                print(
                    f"Visualization error for Epoch {self.current_epoch}, Batch {batch_idx}: {str(e)}"
                )
                self.logger.experiment.log(
                    {
                        "validation_error": f"Epoch {self.current_epoch} Batch {batch_idx}: {str(e)}"
                    }
                )

        # Compute and log metrics
        metrics = {"PSNR": [], "SSIM": [], "MSE": []}
        for i in range(lr_img.shape[0]):
            hr_img_np = hr_img[i].detach().cpu().numpy().transpose(1, 2, 0)
            sr_img_np = sr_img[i].detach().cpu().numpy().transpose(1, 2, 0)
            hr_img_np = (hr_img_np + 1) / 2
            sr_img_np = (sr_img_np + 1) / 2
            hr_img_np = hr_img_np[
                : padding_info["hr"][i][1], : padding_info["hr"][i][0], :
            ]
            sr_img_np = sr_img_np[
                : padding_info["hr"][i][1], : padding_info["hr"][i][0], :
            ]
            psnr = peak_signal_noise_ratio(
                hr_img_np, sr_img_np, data_range=1.0)
            ssim = structural_similarity(
                hr_img_np, sr_img_np, channel_axis=-1, data_range=1.0
            )
            mse = np.mean((hr_img_np - sr_img_np) ** 2)
            metrics["PSNR"].append(psnr)
            metrics["SSIM"].append(ssim)
            metrics["MSE"].append(mse)

        lpips = self.lpips(hr_img, sr_img).cpu().item()
        self.log(
            "val/PSNR",
            np.mean(metrics["PSNR"]),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val/SSIM",
            np.mean(metrics["SSIM"]),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val/MSE",
            np.mean(metrics["MSE"]),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val/LPIPS",
            lpips,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

    def plot_images_with_metrics(
        self,
        hr_img: torch.Tensor,
        lr_img: torch.Tensor,
        sr_img: torch.Tensor,
        padding_info: dict,
        title: str,
        per_image_metrics: list,
    ) -> np.ndarray:
        """Enhanced plotting method with per-image metrics overlaid.

        Plots 5 random triples of HR/LR/SR images with PSNR/SSIM/LPIPS metrics as subtitles.
        Returns a single array for W&B logging.

        Parameters
        ----------
        hr_img : torch.Tensor
            High-resolution images.
        lr_img : torch.Tensor
            Low-resolution images.
        sr_img : torch.Tensor
            Super-resolved images.
        padding_info : dict
            Padding information.
        title : str
            Plot title.
        per_image_metrics : list
            List of (PSNR, SSIM, LPIPS) tuples for each image.

        Returns
        -------
        np.ndarray
            Plotted image array.
        """
        fig, axs = plt.subplots(
            3, 5, figsize=(15, 9), dpi=100
        )  # Adjusted for better resolution
        fig.suptitle(title, fontsize=14, fontweight="bold", color="white")

        # Set background to dark for better contrast with W&B themes
        fig.patch.set_facecolor("#1e1e1e")
        for ax in axs.flat:
            ax.set_facecolor("#2e2e2e")

        for i in range(5):
            num = np.random.randint(0, len(per_image_metrics))
            psnr, ssim, lpips_val = per_image_metrics[num]

            # Normalize and clip images
            sr_img_plot = (
                torch.clamp(sr_img[num], -1,
                            1).cpu().float().numpy().transpose(1, 2, 0)
            )
            sr_img_plot = (sr_img_plot + 1) / 2
            sr_img_plot = np.clip(sr_img_plot, 0, 1)[
                : padding_info["hr"][num][1], : padding_info["hr"][num][0], :
            ]

            hr_img_true = hr_img[num].cpu().float().numpy().transpose(1, 2, 0)
            hr_img_true = (hr_img_true + 1) / 2
            hr_img_true = np.clip(hr_img_true, 0, 1)[
                : padding_info["hr"][num][1], : padding_info["hr"][num][0], :
            ]

            lr_img_true = lr_img[num].cpu().float().numpy().transpose(1, 2, 0)
            lr_img_true = (lr_img_true + 1) / 2
            lr_img_true = np.clip(lr_img_true, 0, 1)[
                : padding_info["lr"][num][1], : padding_info["lr"][num][0], :
            ]

            # Convert to uint8
            sr_img_plot = (sr_img_plot * 255).astype(np.uint8)
            hr_img_true = (hr_img_true * 255).astype(np.uint8)
            lr_img_true = (lr_img_true * 255).astype(np.uint8)

            # Plot with enhanced styling
            axs[0, i].imshow(hr_img_true)
            axs[0, i].set_title("HR Ground Truth", fontsize=9, color="white")
            axs[0, i].set_xticks([])
            axs[0, i].set_yticks([])

            axs[1, i].imshow(lr_img_true)
            axs[1, i].set_title("LR Input", fontsize=9, color="white")
            axs[1, i].set_xticks([])
            axs[1, i].set_yticks([])

            axs[2, i].imshow(sr_img_plot)
            axs[2, i].set_title(
                f"SR Predicted\nPSNR: {psnr:.2f}\nSSIM: {ssim:.3f}\nLPIPS: {lpips_val:.3f}",
                fontsize=7,
                color="limegreen",
                pad=2,
            )
            axs[2, i].set_xticks([])
            axs[2, i].set_yticks([])

        # Tight layout and convert to image
        plt.tight_layout(pad=0.5)
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(
            fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img_array

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict:
        """Test step for the SupResDiffGAN model, for model evaluation.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            Batch of data containing low-resolution and high-resolution images.
        batch_idx : int
            Index of the batch.

        Returns
        -------
        dict
            Dictionary containing the metrics for the batch.
        """
        lr_img, hr_img = batch["lr"], batch["hr"]
        lr_img, hr_img = lr_img.to(self.device), hr_img.to(self.device)
        padding_info = {"lr": batch["padding_data_lr"],
                        "hr": batch["padding_data_hr"]}

        start_time = time.perf_counter()
        sr_img = self(lr_img)
        elapsed_time = time.perf_counter() - start_time

        if batch_idx == 0:
            img_array = self.plot_images_with_metrics(
                hr_img,
                lr_img,
                sr_img,
                padding_info,
                title=f"Test Images: Timesteps {self.diffusion.timesteps}, Posterior {self.diffusion.posterior_type}",
            )
            self.logger.experiment.log(
                {"Test Images": [wandb.Image(
                    img_array, caption="Test Set SR Results")]}
            )

        metrics = {"PSNR": [], "SSIM": [], "MSE": []}
        for i in range(lr_img.shape[0]):
            hr_img_np = hr_img[i].detach().cpu().numpy().transpose(1, 2, 0)
            sr_img_np = sr_img[i].detach().cpu().numpy().transpose(1, 2, 0)
            hr_img_np = (hr_img_np + 1) / 2
            sr_img_np = (sr_img_np + 1) / 2
            hr_img_np = hr_img_np[
                :, : padding_info["hr"][i][1], : padding_info["hr"][i][0]
            ]
            sr_img_np = sr_img_np[
                :, : padding_info["hr"][i][1], : padding_info["hr"][i][0]
            ]
            psnr = peak_signal_noise_ratio(
                hr_img_np, sr_img_np, data_range=1.0)
            ssim = structural_similarity(
                hr_img_np, sr_img_np, channel_axis=-1, data_range=1.0
            )
            mse = np.mean((hr_img_np - sr_img_np) ** 2)
            metrics["PSNR"].append(psnr)
            metrics["SSIM"].append(ssim)
            metrics["MSE"].append(mse)

        lpips = self.lpips(hr_img, sr_img).cpu().item()
        result = {
            "PSNR": np.mean(metrics["PSNR"]),
            "SSIM": np.mean(metrics["SSIM"]),
            "MSE": np.mean(metrics["MSE"]),
            "LPIPS": lpips,
            "time": elapsed_time,
        }
        self.test_step_outputs.append(result)
        return result

    def on_test_epoch_end(self) -> None:
        """Aggregate the metrics for all batches at the end of the test epoch."""
        avg_psnr = np.mean([x["PSNR"] for x in self.test_step_outputs])
        avg_ssim = np.mean([x["SSIM"] for x in self.test_step_outputs])
        avg_mse = np.mean([x["MSE"] for x in self.test_step_outputs])
        avg_lpips = np.mean([x["LPIPS"] for x in self.test_step_outputs])
        avg_time = np.mean([x["time"] for x in self.test_step_outputs])

        self.log(
            "test/PSNR",
            avg_psnr,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "test/SSIM",
            avg_ssim,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "test/MSE",
            avg_mse,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "test/LPIPS",
            avg_lpips,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "test/time",
            avg_time,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.test_step_outputs.clear()

    def discriminator_loss(
        self, hr_s_img: torch.Tensor, sr_s_img: torch.Tensor
    ) -> torch.Tensor:
        """Compute the discriminator loss."""
        y = torch.randint(0, 2, (hr_s_img.shape[0],), device=hr_s_img.device)
        y = y.view(-1, 1, 1, 1)
        y_expanded = y.expand(
            -1, hr_s_img.shape[1], hr_s_img.shape[2], hr_s_img.shape[3]
        )
        first = torch.where(y_expanded == 0, hr_s_img, sr_s_img)
        second = torch.where(y_expanded == 0, sr_s_img, hr_s_img)
        x = torch.cat([first, second], dim=1)
        prediction = self.discriminator(x)
        d_loss = self.adversarial_loss(prediction, y.float())
        self.calculate_ema_noise_step(prediction, y.float())
        return d_loss

    def generator_loss(
        self,
        x0: torch.Tensor,
        x_gen: torch.Tensor,
        hr_img: torch.Tensor,
        sr_img: torch.Tensor,
        hr_s_img: torch.Tensor,
        sr_s_img: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the generator loss."""
        content_loss = self.content_loss(x_gen, x0)
        perceptual_loss = (
            self.vgg_loss(sr_img, hr_img) if self.vgg_loss is not None else 0
        )
        y = torch.randint(0, 2, (hr_s_img.shape[0],), device=hr_s_img.device)
        y = y.view(-1, 1, 1, 1)
        y_expanded = y.expand(
            -1, hr_s_img.shape[1], hr_s_img.shape[2], hr_s_img.shape[3]
        )
        first = torch.where(y_expanded == 0, sr_s_img, hr_s_img)
        second = torch.where(y_expanded == 0, hr_s_img, sr_s_img)
        x = torch.cat([first, second], dim=1)
        prediction = self.discriminator(x)
        adversarial_loss = self.adversarial_loss(prediction, y.float())
        y_reversed = 1 - y.float()
        self.calculate_ema_noise_step(prediction, y_reversed)
        g_loss = (
            content_loss
            + self.alfa_perceptual * perceptual_loss
            + self.alfa_adv * adversarial_loss
        )
        self.log(
            "train/g_content_loss",
            content_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train/g_adv_loss",
            adversarial_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        if self.vgg_loss is not None:
            self.log(
                "train/g_perceptual_loss",
                perceptual_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
        return g_loss

    def configure_optimizers(
        self,
    ) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """Configure optimizers for the generator and discriminator."""
        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr, betas=self.betas
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr, betas=self.betas
        )
        return [opt_g, opt_d]

    def calculate_ema_noise_step(self, pred: torch.Tensor, y: torch.Tensor) -> None:
        """Calculate the Exponential Moving Average (EMA) noise step."""
        pred_binary = (torch.sigmoid(pred) > 0.5).float()
        acc = (pred_binary == y).float().mean().cpu().item()
        self.ema_mean = acc * (1 - self.ema_weight) + \
            self.ema_mean * self.ema_weight
        self.s = int(
            torch.clamp(
                torch.tensor((self.ema_mean - 0.5) * 2 *
                             self.diffusion.timesteps),
                0,
                self.diffusion.timesteps - 1,
            ).item()
        )
        self.log(
            "train/ema_noise_step",
            self.s,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train/ema_accuracy",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )