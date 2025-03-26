# Info

# DataSet

CelebA
http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

source: CelebA/Img/img_align_celeba.zip
save path: data/celeba/img_align_celeba
example: data/celeba/img_align_celeba/000001.jpg

# Env(windows)

CUDA Toolkit 12.8 Update 1
https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local

```
# verify
nvidia-smi

nvcc -V
```

PyTorch (CUDA)

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

# Train

```
python train.py
```

# Test

```
python generate.py -load_path=model/model_default.pth
```