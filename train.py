import torch
import torch.optim as optim
import torch.nn as nn
from utils import evaluate_model
import os

def train_model(model, train_loader, test_loader, device, epochs=5, lr=0.01, model_save_path="models/base_model.pt"):
    """Trains model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    print(f"Starting training on {device}...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

        # Evaluate after each epoch
        accuracy = evaluate_model(model, test_loader, device)
        print(f'Epoch [{epoch+1}/{epochs}] completed. Test Accuracy: {accuracy:.2f}%')

    print("Finished Training.")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")