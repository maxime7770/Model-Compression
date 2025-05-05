import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import os

from utils import evaluate_model


def distillation_loss_fn(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    """
    Calculates the combined loss for knowledge distillation.
    """
    student_loss = F.cross_entropy(student_logits, labels)

    soft_teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    soft_student_log_probs = F.log_softmax(student_logits / temperature, dim=-1) # LogSoftmax for KLDiv
    distill_loss = F.kl_div(
        soft_student_log_probs,
        soft_teacher_probs.detach(),
        reduction='batchmean'
    ) * (temperature ** 2)

    # Combine the two losses
    total_loss = alpha * student_loss + (1 - alpha) * distill_loss
    return total_loss

def train_student_with_distillation(teacher_model, student_model, train_loader, test_loader, device, epochs=5, lr=0.01, temperature=2.0, alpha=0.7, model_save_path="models/student_distilled.pt"):
    """Trains the student model using knowledge distillation."""
    teacher_model.to(device)
    teacher_model.eval() # Teacher model is fixed and used for inference only
    student_model.to(device)

    optimizer = optim.Adam(student_model.parameters(), lr=lr)

    print(f"Starting knowledge distillation training on {device}...")
    print(f"  Teacher: {type(teacher_model).__name__}, Student: {type(student_model).__name__}")
    print(f"  Epochs: {epochs}, LR: {lr}, Temperature: {temperature}, Alpha: {alpha}")

    for epoch in range(epochs):
        student_model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher_model(inputs)

            student_logits = student_model(inputs)

            loss = distillation_loss_fn(student_logits, teacher_logits, labels, temperature, alpha)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

        accuracy = evaluate_model(student_model, test_loader, device)
        print(f'Epoch [{epoch+1}/{epochs}] completed. Student Test Accuracy: {accuracy:.2f}%')

    print("Finished Distillation Training.")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(student_model.state_dict(), model_save_path)
    print(f"Distilled student model saved to {model_save_path}")

