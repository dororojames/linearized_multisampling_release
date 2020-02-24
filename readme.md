# Linearized Multi-Sampling for Differentiable Image Transformation (ICCV 2019)

This repository is a reference implementation for "Linearized Multi-Sampling for Differentiable Image Transformation", ICCV 2019. If you use this code in your research, please cite the paper.

[ArXiv](https://arxiv.org/abs/1901.07124)

### Installation

This implementation is based on Python3 and PyTorch.

### Gradient Visualize

- sample with bilinear

![bilinear](https://github.com/dororojames/linearized_multisampling_release/blob/master/bilinear.png)

- sample with linearized

![bilinear](https://github.com/dororojames/linearized_multisampling_release/blob/master/linearized.png)

### Tutorial

A tutorial is in `main.py` . We built the method to have the same function prototype as `torch.nn.functional.grid_sample`, so you can replace bilinear sampling with linearized multi-sampling with minimum modification.

### Direct plug-in

Copy `sampler.py` to your project folder, and replace `torch.nn.functional.grid_sample` in your code with `sampler.grid_sample`.

### Notes

If you find linearized multi-sampling useful in you project, please feel free to let us know by leaving an issue on this git repository or sending an email to dororojames.cs07g@nctu.edu.tw.
