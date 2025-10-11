# SupResDiffGAN: Super-Resolution Diffusion Generative Adversarial Network

A novel approach for super-resolution tasks that combines diffusion transformers with GANs for high-quality image upscaling.

## Features

- **Diffusion + GAN Architecture**: Combines the power of diffusion models with adversarial training
- **Multiple Variants**: 
  - `SupResDiffGAN`: Full model with discriminator and adversarial loss
  - `SupResDiffGAN_without_adv`: Ablation study without adversarial loss
  - `SupResDiffGAN_simple_gan`: Simplified GAN without Gaussian noise augmentation
- **Local Logging**: All visualizations and metrics saved locally (no external dependencies)
- **TensorBoard Integration**: Standard ML logging with TensorBoard

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements-gpu.txt
   ```

2. **Train Model**:
   ```bash
   python train_model.py
   ```

3. **Evaluate Model**:
   ```bash
   python evaluate_model.py
   ```

4. **View Results**:
   - TensorBoard: `tensorboard --logdir logs/tensorboard`
   - Images: Browse `outputs/` directory
   - Charts: View PNG files in `outputs/charts/`

## Configuration

Edit configuration files in `conf/` directory to customize:
- Model variant selection
- Training parameters
- Dataset settings
- Evaluation metrics

See `CONFIGS.md` for detailed parameter documentation.

