# Related paper: RDDL

Tzu-Hsien Yang*, Zhan-Yi Liao, and Yu-Huai Yu, "A systematic ensemble pipeline for enhancing the identification of lowly prevalent diseases using deep learning". (Submitting)

## Hyperparameters used in Melanoma Detection

### Base Model: Alexnet 

The hyperparameters used for the 10 balancing models are as the following: 

(1) OS (1:1)

```
{
  "name": "MEL-ALEX",
  "batch_size": 64,
  "dropout_rate": 0.42,
  "epochs": 40,
  "initial_lr": 2.4e-7,
  "decay_rate": 0.87,
  "bottom_lr": 1e-12,
  "random_seed": 1182,
}
```

(2) OS (1:2)

```
{
  "name": "MEL-ALEX",
  "batch_size": 64,
  "dropout_rate": 0.45,
  "epochs": 40,
  "initial_lr": 1.8e-7,
  "decay_rate": 0.87,
  "bottom_lr": 1e-12,
  "random_seed": 1182,
}
```

(3) US (1:1)

```
{
  "name": "MEL-ALEX",
  "batch_size": 64,
  "dropout_rate": 0.3,
  "epochs": 40,
  "initial_lr": 5.2e-7,
  "decay_rate": 0.89,
  "bottom_lr": 1e-12,
  "random_seed": 1182,
}
```

(4) US (1:2)

```
{
  "name": "MEL-ALEX",
  "batch_size": 64,
  "dropout_rate": 0.36,
  "epochs": 40,
  "initial_lr": 5.3e-7,
  "decay_rate": 0.88,
  "bottom_lr": 1e-12,
  "random_seed": 1182,
}
```

(5) BB

```
{
  "name": "MEL-ALEX",
  "batch_size": 64,
  "dropout_rate": 0.45,
  "epochs": 40,
  "initial_lr": 2.6e-7,
  "decay_rate": 0.87,
  "bottom_lr": 1e-12,
  "random_seed": 1182,
}
```

(6) CW

```
{
  "name": "MEL-ALEX",
  "batch_size": 64,
  "dropout_rate": 0.28,
  "epochs": 40,
  "initial_lr": 4.2e-7,
  "decay_rate": 0.87,
  "bottom_lr": 1e-12,
  "random_seed": 1182,
}
```

(7) SW

```
{
  "name": "MEL-ALEX",
  "batch_size": 64,
  "dropout_rate": 0.35,
  "epochs": 40,
  "initial_lr": 2.7e-7,
  "decay_rate": 0.88,
  "bottom_lr": 1e-12,
  "random_seed": 1182,
}
```

(8) FL-gamma=1

```
{
  "name": "MEL-ALEX",
  "batch_size": 64,
  "dropout_rate": 0.24,
  "epochs": 40,
  "initial_lr": 3.3e-7,
  "decay_rate": 0.88,
  "bottom_lr": 1e-12,
  "random_seed": 1182,
}
```

(9) FL-gamma=2

```
{
  "name": "MEL-ALEX",
  "batch_size": 64,
  "dropout_rate": 0.47,
  "epochs": 40,
  "initial_lr": 9e-7,
  "decay_rate": 0.87,
  "bottom_lr": 1e-12,
  "random_seed": 1182,
}
```

(10) MFE

```
{
  "name": "MEL-ALEX",
  "batch_size": 64,
  "dropout_rate": 0.25,
  "epochs": 40,
  "initial_lr": 4.7e-7,
  "decay_rate": 0.9,
  "bottom_lr": 1e-12,
  "random_seed": 1182,
}
```

### Base Model: EfficientNet B0

The hyperparameters used for the 10 balancing models are as the following: 

(1) OS (1:1)

```
{
  "name": "MEL-EFF",
  "batch_size": 64,
  "dropout_rate": 0.77,
  "epochs": 40,
  "initial_lr": 1.8e-5,
  "decay_rate": 0.89,
  "bottom_lr": 1e-7,
  "random_seed": 621,
}
```

(2) OS (1:2)

```
{
  "name": "MEL-EFF",
  "batch_size": 64,
  "dropout_rate": 0.74,
  "epochs": 40,
  "initial_lr": 1.5e-5,
  "decay_rate": 0.87,
  "bottom_lr": 1e-7,
  "random_seed": 621,
}
```

(3) US (1:1)

```
{
  "name": "MEL-EFF",
  "batch_size": 64,
  "dropout_rate": 0.36,
  "epochs": 40,
  "initial_lr": 3.4e-5,
  "decay_rate": 0.84,
  "bottom_lr": 1e-7,
  "random_seed": 621,
}
```

(4) US (1:2)

```
{
  "name": "MEL-EFF",
  "batch_size": 64,
  "dropout_rate": 0.36,
  "epochs": 40,
  "initial_lr": 3e-5,
  "decay_rate": 0.86,
  "bottom_lr": 1e-7,
  "random_seed": 621,
}
```

(5) BB

```
{
  "name": "MEL-EFF",
  "batch_size": 64,
  "dropout_rate": 0.78,
  "epochs": 40,
  "initial_lr": 1.8e-5,
  "decay_rate": 0.88,
  "bottom_lr": 1e-7,
  "random_seed": 621,
}
```

(6) CW

```
{
  "name": "MEL-EFF",
  "batch_size": 64,
  "dropout_rate": 0.27,
  "epochs": 40,
  "initial_lr": 2.2e-5,
  "decay_rate": 0.87,
  "bottom_lr": 1e-7,
  "random_seed": 621,
}
```

(7) SW

```
{
  "name": "MEL-EFF",
  "batch_size": 64,
  "dropout_rate": 0.2,
  "epochs": 40,
  "initial_lr": 2.1e-5,
  "decay_rate": 0.84,
  "bottom_lr": 1e-7,
  "random_seed": 621,
}
```

(8) FL-gamma=1

```
{
  "name": "MEL-EFF",
  "batch_size": 64,
  "dropout_rate": 0.26,
  "epochs": 40,
  "initial_lr": 1.2e-5,
  "decay_rate": 0.9,
  "bottom_lr": 1e-7,
  "random_seed": 621,
}
```

(9) FL-gamma=2

```
{
  "name": "MEL-EFF",
  "batch_size": 64,
  "dropout_rate": 0.36,
  "epochs": 40,
  "initial_lr": 1.4e-5,
  "decay_rate": 0.88,
  "bottom_lr": 1e-7,
  "random_seed": 621,
}
```

(10) MFE

```
{
  "name": "MEL-EFF",
  "batch_size": 64,
  "dropout_rate": 0.27,
  "epochs": 40,
  "initial_lr": 1.7e-5,
  "decay_rate": 0.89,
  "bottom_lr": 1e-7,
  "random_seed": 621,
}
```
