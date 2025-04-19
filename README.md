# ConvNext on MLX

Training a smol ConvNext model on CIFAR-10 using the MLX framework. I wrote this mostly as an exercise to learn MLX, and to refresh my memory on ConvNets.

I used the [PyTorch implementation](https://pytorch.org/vision/main/_modules/torchvision/models/convnext.html#convnext_tiny) as my reference; hyperparameters (kernel size in blocks/downsampling etc.) were adapted for CIFAR-10 based on [this implementation by Julius Ruseckas](https://juliusruseckas.github.io/ml/convnext-cifar10.html) and [this GitHub issue](https://github.com/facebookresearch/ConvNeXt/issues/134#issuecomment-1534986992).

The training loop code is in `main.py`, and is exactly the same as the [training loop for Resnets](https://github.com/ml-explore/mlx-examples/blob/main/cifar/main.py) in the mlx-examples repo.

## Pre-requisites

Follow the instructions on the [mlx-examples](https://github.com/ml-explore/mlx) repository.

## Running the example

Run the example with:

```
python main.py
```

By default the example runs on the GPU. To run on the CPU, use: 

```
python main.py --cpu
```

For all available options, run:

```
python main.py --help
```

## Results

After training with the default `ConvNeXt_Smol` architecture for 30 epochs, I got the following results:

```
Epoch: 29 | avg. Train loss 0.451 | avg. Train acc 0.841 | Throughput: 2291.73 images/sec
Epoch: 29 | Test acc 0.802
```

Note this was run on an M1 Pro Macbook Pro with 32GB RAM.

Why isn't it training as fast as, or faster than `resnet20`, even though `ConvNeXt_Smol` is almost 10 times larger? I'm not sure! Drop an issue if you see any obvious/egregious errors in my code.