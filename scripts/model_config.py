# In scripts/model_config.py

from .model_config_imports import *
from diffusers import AutoencoderTiny # Import AutoencoderTiny


def model_selection(cfg, device):
    """Select and initialize SupResDiffGAN model variants based on the configuration.

    Parameters
    ----------
    cfg : DictConfig
        The configuration object containing model parameters and settings.
    device : torch.device
        The device on which the model will be loaded (e.g., 'cuda' or 'cpu').

    Returns
    -------
    torch.nn.Module
        The initialized SupResDiffGAN model.

    Raises
    ------
    ValueError
        If the specified model is not a supported SupResDiffGAN variant.
    """

    if cfg.model.name == "SupResDiffGAN":
        return initialize_supresdiffgan(
            cfg, device, SupResDiffGAN, use_discriminator=True
        )

    elif cfg.model.name == "SupResDiffGAN_without_adv":
        return initialize_supresdiffgan(
            cfg, device, SupResDiffGAN_without_adv, use_discriminator=False
        )

    elif cfg.model.name == "SupResDiffGAN_simple_gan":
        return initialize_supresdiffgan(
            cfg, device, SupResDiffGAN_simple_gan, use_discriminator=True
        )

    else:
        raise ValueError(
            f"Model '{cfg.model.name}' not found. "
            f"Supported models: SupResDiffGAN, SupResDiffGAN_without_adv, SupResDiffGAN_simple_gan"
        )


def initialize_supresdiffgan(cfg, device, model_class, use_discriminator=True):
    """Helper function to initialize SupResDiffGAN variants.

    Parameters
    ----------
    cfg : DictConfig
        The configuration object containing model parameters and settings.
    device : torch.device
        The device on which the model will be loaded (e.g., 'cuda' or 'cpu').
    model_class : class
        The class of the model to initialize (e.g., SupResDiffGAN, SupResDiffGAN_without_adv).
    use_discriminator : bool, optional
        Whether to include the discriminator in the model initialization.
    use_vgg_loss : bool, optional
        Whether to include the VGG loss in the model initialization.

    Returns
    -------
    torch.nn.Module
        The initialized model.
    """
    if cfg.autoencoder == "VAE":
        # Use AutoencoderTiny for lower memory consumption
        model_id = "madebyollin/taesd"  # Model ID for AutoencoderTiny
        autoencoder = AutoencoderTiny.from_pretrained(model_id).to(device)

    discriminator = (
        Discriminator_supresdiffgan(
            in_channels=cfg.discriminator.in_channels,
            channels=cfg.discriminator.channels,
        )
        if use_discriminator
        else None
    )

    unet = UNet_supresdiffgan(cfg.unet)

    diffusion = Diffusion_supresdiffgan(
        timesteps=cfg.diffusion.timesteps,
        beta_type=cfg.diffusion.beta_type,
        posterior_type=cfg.diffusion.posterior_type,
    )

    if cfg.use_perceptual_loss:
        if cfg.feature_extractor:
            vgg_loss = FeatureExtractor_supresdiffgan(device)
        else:
            vgg_loss = VGGLoss_supresdiffgan(device)
    else:
        vgg_loss = None

    if cfg.model.load_model is not None:
        model_path = cfg.model.load_model
        _, ext = os.path.splitext(model_path)
        if ext == ".pth":
            model = model_class(
                ae=autoencoder,
                discriminator=discriminator,
                unet=unet,
                diffusion=diffusion,
                learning_rate=cfg.model.lr,
                alfa_perceptual=cfg.model.alfa_perceptual,
                alfa_adv=cfg.model.alfa_adv,
                vgg_loss=vgg_loss,
            )
            model.load_state_dict(torch.load(model_path, map_location=device))
        elif ext == ".ckpt":
            model = model_class.load_from_checkpoint(
                model_path,
                map_location=device,
                ae=autoencoder,
                discriminator=discriminator,
                unet=unet,
                diffusion=diffusion,
                learning_rate=cfg.model.lr,
                alfa_perceptual=cfg.model.alfa_perceptual,
                alfa_adv=cfg.model.alfa_adv,
                vgg_loss=vgg_loss,
            )

        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    else:
        model = model_class(
            ae=autoencoder,
            discriminator=discriminator,
            unet=unet,
            diffusion=diffusion,
            learning_rate=cfg.model.lr,
            alfa_perceptual=cfg.model.alfa_perceptual,
            alfa_adv=cfg.model.alfa_adv,
            vgg_loss=vgg_loss,
        )

    return model