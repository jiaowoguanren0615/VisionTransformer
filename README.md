<h1 align='center'>VisionTransformer</h1>

### This is a warehouse for ViT-Pytorch-model, can be used to train your image-datasets for vision tasks.  

### [KAN: Kolmogorov–Arnold Networks](https://arxiv.org/abs/2404.19756)  
### [PaLI: A Jointly-Scaled Multilingual Language-Image Model](https://arxiv.org/abs/2209.06794v4)  
### [Scaling Vision Transformers](https://openaccess.thecvf.com//content/CVPR2022/papers/Zhai_Scaling_Vision_Transformers_CVPR_2022_paper.pdf)  
### [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)  
### [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://proceedings.mlr.press/v80/shazeer18a/shazeer18a.pdf)   

![image](https://github.com/jiaowoguanren0615/VisionTransformer/blob/main/sample_png/KAN-model.jpg)  
![image](https://production-media.paperswithcode.com/methods/Screen_Shot_2021-01-26_at_9.43.31_PM_uI4jjMq.png)  
![image](https://github.com/jiaowoguanren0615/VisionTransformer/blob/main/sample_png/vit-family-table.jpg)  
![image](https://github.com/jiaowoguanren0615/VisionTransformer/blob/main/sample_png/vit-e_architechture.jpg)  

## Preparation
### Create conda virtual-environment
```bash
conda env create -f environment.yml
```

### Download the dataset: 
[flower_dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition).  

## Project Structure
```
├── datasets: Load datasets
    ├── my_dataset.py: Customize reading data sets and define transforms data enhancement methods
    ├── split_data.py: Define the function to read the image dataset and divide the training-set and test-set
    ├── threeaugment.py: Additional data augmentation methods
├── models: VisionTransformer Model
    ├── build_models.py: Construct VisionTransformer models
    ├── kan_model.py: Define KAN model replaces MLP layers
├── scheduler:
    ├──scheduler_main.py: Fundamental Scheduler module
    ├──scheduler_factory.py: Create lr_scheduler methods according to parameters what you set
    ├──other_files: Construct lr_schedulers (cosine_lr, poly_lr, multistep_lr, etc)
├── util:
    ├── engine.py: Function code for a training/validation process
    ├── LBFGS_optimizer.py: Define LBFGS_optimizer for training KAN Network  
    ├── losses.py: Knowledge distillation loss, combined with teacher model (if any)
    ├── lr_decay.py: Define "inverse_sqrt_lr_decay" function for "Adafactor" optimizer
    ├── lr_sched.py: Define "adjust_learning_rate" function
    ├── optimizer.py: Define Sophia & Adafactor optimizer(for ViT-e & PaLI-17B)
    ├── samplers.py: Define the parameter of "sampler" in DataLoader
    ├── utils.py: Record various indicator information and output and distributed environment
├── estimate_model.py: Visualized evaluation indicators ROC curve, confusion matrix, classification report, etc.
└── train_gpu.py: Training model startup file (including infer process)
```

## Precautions
Before you use the code to train your own data set, please first enter the ___train_gpu.py___ file and modify the ___data_root___, ___batch_size___,  ___data_len___, ___num_workers___ and ___nb_classes___ parameters. If you want to draw the confusion matrix and ROC curve, you only need to set the ___predict___ parameter to __True__.  

## Use Adafactor Optimizer for training vit-e
You can use anther optimizer Adafactor, just need to change the optimizer in ___train_gpu.py___.
![image](https://github.com/jiaowoguanren0615/VisionTransformer/blob/main/sample_png/Adafactor_vit-e.jpg)

## Train this model

### Parameters Meaning:
```
1. nproc_per_node: <The number of GPUs you want to use on each node (machine/server)>
2. CUDA_VISIBLE_DEVICES: <Specify the index of the GPU corresponding to a single node (machine/server) (starting from 0)>
3. nnodes: <number of nodes (machine/server)>
4. node_rank: <node (machine/server) serial number>
5. master_addr: <master node (machine/server) IP address>
6. master_port: <master node (machine/server) port number>
```
### Transfer Learning:
Step 1: Write the ___pre-training weight path___ into the ___args.fintune___ in string format.  
Step 2: Modify the ___args.freeze_layers___ according to your own GPU memory. If you don't have enough memory, you can set this to True to freeze the weights of the remaining layers except the last layer of classification-head without updating the parameters. If you have enough memory, you can set this to False and not freeze the model weights.  

#### Here is an example for setting parameters:
![image](https://github.com/jiaowoguanren0615/VisionTransformer/blob/main/sample_png/transfer_learning.jpg)


### Note: 
If you want to use multiple GPU for training, whether it is a single machine with multiple GPUs or multiple machines with multiple GPUs, each GPU will divide the batch_size equally. For example, batch_size=4 in my train_gpu.py. If I want to use 2 GPUs for training, it means that the batch_size on each GPU is 4. ___Do not let batch_size=1 on each GPU___, otherwise BN layer maybe report an error.  

### train model with single-machine single-GPU：
```
python train_gpu.py
```

### train model with single-machine multi-GPU：
```
python -m torch.distributed.run --nproc_per_node=8 train_gpu.py
```

### train model with single-machine multi-GPU: 
(using a specified part of the GPUs: for example, I want to use the second and fourth GPUs)
```
CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.run --nproc_per_node=2 train_gpu.py
```

### train model with multi-machine multi-GPU:
(For the specific number of GPUs on each machine, modify the value of --nproc_per_node. If you want to specify a certain GPU, just add CUDA_VISIBLE_DEVICES= to specify the index number of the GPU before each command. The principle is the same as single-machine multi-GPU training)
```
On the first machine: python -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py

On the second machine: python -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=<Master node IP address> --master_port=<Master node port number> train_gpu.py
```

## Citation
```
@article{chen2022pali,
  title={Pali: A jointly-scaled multilingual language-image model},
  author={Chen, Xi and Wang, Xiao and Changpinyo, Soravit and Piergiovanni, AJ and Padlewski, Piotr and Salz, Daniel and Goodman, Sebastian and Grycner, Adam and Mustafa, Basil and Beyer, Lucas and others},
  journal={arXiv preprint arXiv:2209.06794},
  year={2022}
}
```

```
@inproceedings{zhai2022scaling,
  title={Scaling vision transformers},
  author={Zhai, Xiaohua and Kolesnikov, Alexander and Houlsby, Neil and Beyer, Lucas},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={12104--12113},
  year={2022}
}
```

```
@article{dosovitskiy2020image,
  title={An image is worth 16x16 words: Transformers for image recognition at scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```

```
@inproceedings{shazeer2018adafactor,
  title={Adafactor: Adaptive learning rates with sublinear memory cost},
  author={Shazeer, Noam and Stern, Mitchell},
  booktitle={International Conference on Machine Learning},
  pages={4596--4604},
  year={2018},
  organization={PMLR}
}
```
