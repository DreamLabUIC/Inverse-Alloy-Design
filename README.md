# Inverse Alloy Design Tools

This repository provides tools for inverse alloy design using both **gradient-based optimization within the latent space** and the **latent diffusion model**. These models are trained on the **ANSYS/GRANTA dataset**, which cannot be provided due to proprietary restrictions (owned by NASA). However, we have included the trained model files to facilitate easy use by others.

## Features
- **Gradient-Based Optimization**: Optimize compositions within the latent space to meet specific target properties.
- **Latent Diffusion Model**: A generative model that explores the compositional design space efficiently while maintaining physical feasibility.
- **Pre-Trained Models**: The trained models are provided to help users apply the tools without the need for re-training.
- **Compositional Design Exploration**: Efficiently explore alloy compositions and their corresponding properties.

## Requirements
The following Python libraries are required:
- **Pandas** (for data manipulation)
- **NumPy** (for numerical operations)
- **PyTorch** (for model training and inference)

You can install the required libraries using `pip`:

```bash
pip install pandas numpy torch

## License
This code is released for academic and research purposes. Please refer to the repository for specific usage terms.

You can copy and paste the above directly into your GitHub repository's `README.md` file. Let me know if you'd like any further changes!
