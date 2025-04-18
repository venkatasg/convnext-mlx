# ConvNext on MLX

Training a smol ConvNext model on CIFAR-10 using the MLX framework. I wrote this mostly as an exercise to learn MLX, and to refresh my memory on ConvNets.

I used the [PyTorch implementation] as my reference; hyperparameters (kernel size in blocks/downsampling etc.) were adapted for CIFAR-10 based on [this implementation by Julius Ruseckas](https://juliusruseckas.github.io/ml/convnext-cifar10.html) and [this GitHub issue]().



## Pre-requisites

Follow the instructions on the [mlx-examples]() repository.

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

After training with the default `ConvNeXt_Smol` architecture for 10 epochs, you
should see the following results:

```
Epoch: 9 | avg. Train loss 0.769 | avg. Train acc 0.730 | Throughput: 1270.30 images/sec
Epoch: 9 | Test acc 0.724
```

Note this was run on an M1 Pro Macbook Pro with 32GB RAM.
