# Vision Weaver: A SupResDiffGAN Implementation

<div align="center">
  <p><strong>Fusing Diffusion Transformers and GANs for High-Fidelity Super-Resolution</strong></p>
</div>

<div align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python" alt="Python"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch" alt="PyTorch"></a>
  <a href="https://www.pytorchlightning.ai/"><img src="https://img.shields.io/badge/PyTorch%20Lightning-2.2-792ee5?logo=pytorch-lightning" alt="PyTorch Lightning"></a>
  <a href="https://hydra.cc/"><img src="https://img.shields.io/badge/Config-Hydra-89b8cd" alt="Hydra"></a>
  <a href="https://wandb.ai/"><img src="https://img.shields.io/badge/Logged-W%26B-yellowgreen" alt="Weights & Biases"></a>
</div>

Vision Weaver is a high-quality implementation of the research paper *"SupResDiffGAN: A New Approach for the Super-Resolution Task"* by Kopeć et al. [cite: SupResDiffGAN-Paper.pdf]. This project leverages a hybrid architecture that combines the strengths of Denoising Diffusion Models and Generative Adversarial Networks (GANs) to perform image super-resolution. By operating in the latent space of a pre-trained autoencoder, the model achieves a compelling balance between the exceptional perceptual quality of diffusion models and the inference speed of GANs.

This implementation is built using **PyTorch** and **PyTorch Lightning**, with configuration managed by **Hydra** and experiment tracking integrated with **Weights & Biases**.

## 🌟 Results Showcase
Below is an example of the model's performance, upscaling a low-resolution image from the CelebA dataset after being trained.

<!-- <p align="center">
  Left: Low-Resolution Input &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp; Right: Vision Weaver Super-Resolution Output
</p> -->

## 📋 Table of Contents
- ✨ Key Features
- 🛠️ Getting Started
  - Prerequisites
  - Installation
- 📂 Datasets
- ▶️ Usage
  - Training
  - Evaluation
- ⚙️ Configuration
- 🏢 Model Architecture
- 🙏 Acknowledgement
- 📜 License

## ✨ Key Features
- **Hybrid Generative Model**: Implements the SupResDiffGAN architecture, which fuses a U-Net based diffusion generator with a patch-based adversarial discriminator for state-of-the-art results.
- **Latent Space Diffusion**: Performs the computationally intensive diffusion process in the compressed latent space of a tiny autoencoder, enabling fast training and inference.
- **Flexible Configuration**: Utilizes Hydra for comprehensive experiment management. All model, trainer, and dataset parameters are defined in easily editable `.yaml` files.
- **Experiment Tracking**: Integrated with Weights & Biases (wandb) for real-time logging of metrics, validation images, and performance charts.
- **Modular Codebase**: Built with PyTorch Lightning, ensuring a clean separation of model logic (`SupResDiffGAN.py`), data handling (`scripts/data_loader.py`), and training/evaluation scripts (`train_model.py`, `evaluate_model.py`).

## 🛠️ Getting Started

### Prerequisites
- **Python 3.9+**
- An NVIDIA GPU with CUDA support is highly recommended for training.
- A Kaggle account and `kaggle.json` API token for downloading datasets.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/samarthya04/Super-Resolution-Diff-GAN.git
   cd Super-Resolution-Diff-GAN
   ```
2. Create and activate a virtual environment:
   ```bash
   conda create -n SupResDiffGAN_env python=3.10
   conda activate SupResDiffGAN_env
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements-data.txt
   pip install -r requirements-gpu.txt
   ```
4. Log in to Weights & Biases:
   ```bash
   wandb login
   ```
   You will be prompted to enter your API key.

## 📂 Datasets
The project includes a shell script to automate the download and preparation of common super-resolution datasets.

- **CelebA-HQ**: This is the primary dataset used for training the provided configurations.
- Other datasets like ImageNet, DIV2K, etc., are also supported.

To download the CelebA dataset, run the following command. You will need to have your `kaggle.json` file placed in `~/.kaggle/` (for Linux) or `C:\Users\<Your-Username>\.kaggle\` (for Windows).
```bash
bash get_data.sh -c
```

## ▶️ Usage

### Training
The project is configured for a high-quality training run that is optimized for modern GPUs. To start training with the recommended "Maximum Quality" configuration, run the following command:
```bash
python train_model.py -cn "config_supresdiffgan"
```
- Progress will be displayed in the terminal and logged to your Weights & Biases project.
- The best performing model checkpoint will be saved in the `models/checkpoints/` directory, based on the `val/LPIPS` metric.

### Evaluation
After training is complete, you can evaluate your best model on the test set.

1. Update the evaluation config: Open `conf/config_supresdiffgan_evaluation.yaml` and ensure that the `load_model` path points to your best checkpoint file (e.g., `models/checkpoints/SupResDiffGAN_Max_Quality-epoch=XX-val_LPIPS=0.XX.ckpt`).
2. Run the evaluation script:
   ```bash
   python evaluate_model.py
   ```
- The script will run a comprehensive evaluation with different samplers and step counts, as defined in the config.
- Results will be saved to a `.csv` file in the `evaluation_results/` directory, and performance charts will be logged to Weights & Biases.

## ⚙️ Configuration
All aspects of the project are controlled by `.yaml` files located in the `conf/` directory. The main configuration entry point is `config.yaml`, which can be overridden by more specific files.

- `config_supresdiffgan.yaml`: The recommended file for training a high-performance model.
- `config_supresdiffgan_evaluation.yaml`: A pre-configured file for running final model evaluation.

You can modify these files or create new ones to experiment with different hyperparameters, such as:
- Model architecture (unet, discriminator channels)
- Batch size and learning rate
- Loss weights (`alfa_adv`, `alfa_perceptual`)
- Diffusion timesteps

## 🏢 Model Architecture
The Vision Weaver implementation follows the SupResDiffGAN architecture:
- **Encoder**: A pre-trained VAE encoder compresses the low-resolution and high-resolution images into latent representations.
- **U-Net Generator**: A U-Net model performs a denoising diffusion process in the latent space to generate a super-resolved latent.
- **Decoder**: The VAE decoder transforms the super-resolved latent back into a high-resolution image in pixel space.
- **Discriminator**: A GAN discriminator is trained to distinguish between the generated images and real high-resolution images, pushing the generator to create more realistic details.

## 🙏 Acknowledgement
This project is an implementation of the paper *"SupResDiffGAN: A New Approach for the Super-Resolution Task"* by Dawid Kopeć, Wojciech Kozłowski, Maciej Wizerkaniuk, Dawid Krutul, Jan Kocoń, and Maciej Zięba. We are grateful for their foundational research, which made this work possible.

## 📜 License
This project is licensed under the terms of the MIT License. See the [LICENSE](LICENSE) file for the full text.