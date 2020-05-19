# OMLET: Open Machine Learning Engineering Toolbox

## Installation


```bash
git clone https://github.com/LinxiFan/omlet.git
cd omlet
pip install -e .
```


### PyTorch-Lightning

Tested only with v0.7.6, please upgrade.

### Hydra

Due to incompatibilities, please install `hydra` management library from their master branch:

```bash
git clone https://github.com/facebookresearch/hydra.git
cd hydra
pip install -e .
cd plugins/hydra_colorlog
pip install -e .
```

The above should also install OmegaConf 2.0.1, which we'll use as our primary dot-access dict for hyperparameters.

OmegaConf docs: https://omegaconf.readthedocs.io/en/latest/usage.html

### Nvidia APEX fp16 training

https://github.com/NVIDIA/apex#quick-start

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
