# PC-NBV: A Point Cloud Based Deep Network for Efficient Next Best View Planning

### Introduction 

This repository is for our IROS 2020 paper "PC-NBV: A Point Cloud Based Deep Network for Efficient Next Best View Planning". The code is modified from [pcn](https://github.com/wentaoyuan/pcn) and [PU-GAN](https://github.com/liruihui/PU-GAN). 

### Installation
This repository is based on Tensorflow and the TF operators from PointNet++. Therefore, you need to install tensorflow and compile the TF operators. 

For installing tensorflow, please follow the official instructions in [here](https://www.tensorflow.org/install/install_linux). The code is tested under Tensorflow 1.12 (higher version should also work) and Python 3.7 on Ubuntu 16.04.

For compiling TF operators, please check `tf_xxx_compile.sh` under each op subfolder in `tf_ops` folder. Note that you need to update `nvcc`, `python` and `tensoflow include library` if necessary. 

### Note
When running the code, if you have `undefined symbol: _ZTIN10tensorflow8OpKernelE` error, you need to compile the TF operators. If you have already added the `-I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework` but still have ` cannot find -ltensorflow_framework` error. Please use 'locate tensorflow_framework' to locate the tensorflow_framework library and make sure this path is in `$TF_LIB`.

### Usage

1. Clone the repository:

   ```shell
   https://github.com/Smile2020/PC-NBV.git
   cd PC-NBV
   ```
   
2. Compile the TF operators:

   Follow the above information to compile the TF operators. 
   
3. Generate the data:

    First you have to generate networks' inputs and supervision data  and sample complete point cloud data from your model datasets. You can use lmdb_write_shapenet.py to make lmdb data and put them in folder `data`.

4. Train the model:
   ```shell
   python train.py 
   ```

5. Evaluate the model:

   To test your trained model, you can run:
   ```shell
   python test.py --checkpoint model_path
   ```

### Questions

Please contact 'zengr17@mails.tsinghua.edu.cn'

