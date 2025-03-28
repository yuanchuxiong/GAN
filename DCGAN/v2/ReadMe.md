# Info

# DataSet

**CelebA**

http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

source: CelebA/Img/img_align_celeba.zip
save path: data/celeba/img_align_celeba
example: data/celeba/img_align_celeba/000001.jpg

**cifar-10**

https://www.cs.toronto.edu/~kriz/cifar.html

https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

project_root_path/data/cifar-10-python.tar.gz
project_root_path/data/cifar-10-python/cifar-10-batches-py/data_batch_1
project_root_path/data/cifar-10-python/cifar-10-batches-py/data_batch_2
project_root_path/data/cifar-10-python/cifar-10-batches-py/data_batch_3
project_root_path/data/cifar-10-python/cifar-10-batches-py/data_batch_4
project_root_path/data/cifar-10-python/cifar-10-batches-py/data_batch_5
project_root_path/data/cifar-10-python/cifar-10-batches-py/test_batch

extract

```
cd project_root_path
mkdir -p data/cifar-10/train

python cifar10.py
```

project_root_path/data/cifar-10/train/0/xxx.jpg


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
python train.py -dataset='celeba'

python train.py -dataset='cifar-10'
```

# Test

```
python generate.py -load_path=model/model_default.pth
```