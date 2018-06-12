# code for "Improving GAN Training via Binarized Representation Entropy (BRE) Regularization" (ICLR2018 paper)

paper: https://arxiv.org/abs/1805.03644

## Getting data

### For cifar10
Get the matlab version of cifar10 from https://www.cs.toronto.edu/~kriz/cifar.html
run `python utils/preprocess_cifar10.py [path_to_downloaded_file]`

### For celeba
Get the Align&Cropped version of celeba from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
run `python utils/preprocess_celeba.py [path_to_downloaded_file]`

## Dependencies
 tensorflow (need 1.4.0 for the evaluation code to run properly)
 lasagne (tested with 0.2.dev1)
 theano (tested with 1.0.1)
 cuda (tested with version 8.0)
 cudnn (tested with 7005)
 
 Version probably doens't matter: 
 joblib
 scipy
 numpy
 PIL

## Train GAN:
To run a default experiment (version of dcgan used in paper, bre turned on, cifar10 experiments)
``` bash
python run.py 
```

To reproduce results from the paper for DCGAN equal size for example:
``` bash
python run.py max_iteration:40000 bre_w:1 model_func:dcgan_equal_size monitor:1
```

To turn off BRE for baseline:
``` bash
python run.py max_iteration:40000 bre_w:0 model_func:dcgan_equal_size monitor:1
```

## Note:
FID scoring is added, which was not used and reported at the time of publication



