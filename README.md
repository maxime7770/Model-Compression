# Model Compression Demo

Showcases pruning, quantization, low-rank factorization, and knowledge distillation on a simple CIFAR-10 CNN model.
This repository is associated with my blog post on model compression techniques. Check it out here: [Model Compression Techniques](https://towardsdatascience.com/model-compression-make-your-machine-learning-models-lighter-and-faster/).

## Prerequisites

- Python 3.x
- Install dependencies with: `pip install -r requirements.txt`

## Usage

- To run all steps:  
  `python main.py`

## Outputs

- Generated plots: `model_sizes.png`, `model_latencies.png`, `model_accuracy.png`
- Model checkpoints saved in `models/`