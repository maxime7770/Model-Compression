import torch
import time
import os
import copy

def evaluate_model(model, data_loader, device):
    """Evaluates the model's accuracy on the given data loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def measure_latency(model, device, input_size=(1, 3, 32, 32), iterations=100):
    """Measures the average inference latency of the model"""
    model.eval()
    dummy_input = torch.randn(*input_size, device=device)
    
    # warm up runs
    with torch.no_grad():
        for _ in range(10):
            model(dummy_input)

    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            model(dummy_input)
    end_time = time.time()
    
    avg_latency = (end_time - start_time) / iterations * 1000 # milliseconds
    return avg_latency

def get_model_size(model, filepath="temp_model.pt"):
    """Calculates the model size in MB."""
    model_cpu = copy.deepcopy(model).to('cpu')
    torch.save(model_cpu.state_dict(), filepath)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    os.remove(filepath) # Clean up the temp file
    return size_mb

def count_parameters(model):
    """Counts the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_stats(model, model_name, test_loader, device):
     accuracy = evaluate_model(model, test_loader, device)
     latency = measure_latency(model, device)
     size_mb = get_model_size(model)
     params = count_parameters(model)
     print(f"--- {model_name} ---")
     print(f"  Accuracy: {accuracy:.2f}%")
     print(f"  Size (MB): {size_mb:.4f}")
     print(f"  Latency (ms): {latency:.4f}")
     print(f"  Parameters: {params}")
     print("-" * (len(model_name) + 8))
     return {"accuracy": accuracy, "size_mb": size_mb, "latency": latency, "params": params}

