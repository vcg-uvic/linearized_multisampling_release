# Linearized Multi-Sampling for Differentiable Image Transformation (ICCV 2019)

This repository is a reference implementation for "Linearized Multi-Sampling for Differentiable Image Transformation", ICCV 2019. If you use this code in your research, please cite the paper.

[ArXiv](https://arxiv.org/abs/1901.07124)

### Installation

This implementation is based on Python3 and PyTorch.

You can install the environment by: ```conda env create -f environment.yml```

Activate the env by: ```conda activate linearized```

### Tutorial

A tutorial is in `linearized sampler tutorial.ipynb` . We built the method to have the same function prototype as `torch.nn.functional.grid_sample`, so you can replace bilinear sampling with linearized multi-sampling with minimum modification.

### Direct plug-in

Copy `./warp/linearized.py` to your project folder, and replace `torch.nn.functional.grid_sample` in your code with `linearized.grid_sample`. 

We made `linearize.py` to have minimum dependencies(PyTorch only), so we put some extra utils methods in that file. You can move those utils methods to another place to make it cleaner.

### Notes

If you find linearized multi-sampling useful in you project, please feel free to let us know by leaving an issue on this git repository or sending an email to jiangwei@uvic.ca.
