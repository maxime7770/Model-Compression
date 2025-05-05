import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from data import get_data_loaders
from model import CIFAR10Model, CIFAR10StudentModel
from train import train_model
from pruning import apply_pruning
from quantization import apply_quantization
from low_rank_factorization import apply_low_rank_factorization
from knowledge_distillation import train_student_with_distillation
from utils import evaluate_model, get_model_size, measure_latency, count_parameters
import copy
import matplotlib.pyplot as plt
import os

device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_loader, test_loader, calib_loader = get_data_loaders()

model = CIFAR10Model().to(device)
BASE_MODEL_PATH = "models/cifar10_base.pt"
if os.path.exists(BASE_MODEL_PATH):
    print(f"Loading existing base model from {BASE_MODEL_PATH}")
    model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=device))
else:
    train_model(model, train_loader, test_loader, device, epochs=10, lr=1e-3, model_save_path=BASE_MODEL_PATH)
print("Base Accuracy:", evaluate_model(model, test_loader, device))
params_base = count_parameters(model)
print(f"Base Model Params: {params_base}")

# 2. Pruning
pruned_model = apply_pruning(model, amount=0.15, min_channels=128)
pruned_model.to(device)
acc_pruned = evaluate_model(pruned_model, test_loader, device)
params_pruned = count_parameters(pruned_model)
print(f"Pruned Model Params: {params_pruned}")
size_pruned = get_model_size(pruned_model)
lat_pruned = measure_latency(pruned_model, device, input_size=(1,3,32,32), iterations=1000)
print(f"Pruned Model Acc: {acc_pruned:.2f}% / Size: {size_pruned:.4f} MB / Latency: {lat_pruned:.2f} ms")

# 3. Static Post-Training Quantization (PTQ)
q_model = apply_quantization(copy.deepcopy(model), calib_loader)
acc_q = evaluate_model(q_model, test_loader, torch.device("cpu")) # on macos, using GPU causes issues
params_q = count_parameters(q_model)
print(f"Quantized Model Params: {params_q}")
size_q = get_model_size(q_model)
lat_q = measure_latency(q_model, torch.device("cpu"), input_size=(1,3,32,32), iterations=1000)
print(f"Quantized Model Acc: {acc_q:.2f}% / Size: {size_q:.4f} MB / Latency: {lat_q:.2f} ms")

# 4. Low-Rank Factorization
lr_model = apply_low_rank_factorization(model, fraction=0.4)
lr_model.to(device)
acc_lr = evaluate_model(lr_model, test_loader, device)
params_lr = count_parameters(lr_model)
print(f"Low-Rank Model Params: {params_lr}")
size_lr = get_model_size(lr_model)
lat_lr = measure_latency(lr_model, device, input_size=(1,3,32,32), iterations=1000)
print(f"Low-Rank Model Acc: {acc_lr:.2f}% / Size: {size_lr:.4f} MB / Latency: {lat_lr:.2f} ms")

# 5. Knowledge Distillation
student = CIFAR10StudentModel().to(device)
train_student_with_distillation(model, student, train_loader, test_loader, device, epochs=8, lr=1e-3)
acc_stu = evaluate_model(student, test_loader, device)
params_stu = count_parameters(student)
print(f"Distilled Student Params: {params_stu}")
size_stu = get_model_size(student)
lat_stu = measure_latency(student, device, input_size=(1,3,32,32), iterations=1000)
print(f"Distilled Student Acc: {acc_stu:.2f}% / Size: {size_stu:.4f} MB / Latency: {lat_stu:.2f} ms")


######## Comparison of all methods ########

base = CIFAR10Model().to(device)
torch.load_state_dict = torch.load
base.load_state_dict(torch.load("models/cifar10_base.pt", map_location=device))

pruned = apply_pruning(copy.deepcopy(base), amount=0.15, min_channels=128)
stat = apply_quantization(copy.deepcopy(base).cpu(), calib_loader)
lr = apply_low_rank_factorization(copy.deepcopy(base), fraction=0.4)
student = student  # last trained student (see above)

variants = {"Base": base, "Pruned": pruned, "Quantized": stat, "LowRank": lr, "Distilled": student}

# Compute metrics on CPU for comparison
sizes = {k: get_model_size(m) for k,m in variants.items()}
latencies = {k: measure_latency(m.cpu(), torch.device('cpu'), input_size=(1,3,32,32), iterations=10000)
             for k,m in variants.items()}
accs = {k: evaluate_model(m.to(device if k!='Quantized' else torch.device('cpu')), test_loader,
                        torch.device('cpu') if k=='Quantized' else device)
        for k,m in variants.items()}

plt.figure()
plt.bar(sizes.keys(), sizes.values(), color='skyblue')
plt.ylabel('Size (MB)')
plt.title('Model Size Comparison')
plt.savefig('model_sizes.png')

plt.figure()
plt.bar(latencies.keys(), latencies.values(), color='salmon')
plt.ylabel('Latency (ms)')
plt.title('Inference Latency Comparison')
plt.savefig('model_latencies.png')

plt.figure()
plt.bar(accs.keys(), accs.values(), color='lightgreen')
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy Comparison')
plt.savefig('model_accuracy.png')

print("Saved comparison plots: model_sizes.png, model_latencies.png, model_accuracy.png")
