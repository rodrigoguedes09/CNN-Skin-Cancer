# main.py
import torch
import torch.optim as optim
from data.balanced_data_generator import create_data_generators
from src.model import create_model, get_training_parameters

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Cria os data loaders balanceados com augmentation para malignos
    train_loader, val_loader = create_data_generators()
    
    # Cria e move modelo para GPU
    model = create_model()
    model = model.to(device)
    
    criterion, optimizer_params = get_training_parameters()
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), **optimizer_params)
    
    num_epochs = 20
    best_val_acc = 0.0
    
    # Imprime cabeçalho da tabela de métricas
    print("Epoch\tTrain Loss\tTrain Acc\tVal Loss\tVal Acc")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # ---------------------------
        # Treinamento
        # ---------------------------
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        
        # ---------------------------
        # Validação
        # ---------------------------
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.float().to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                running_val_loss += loss.item()
                predicted = (outputs.squeeze() > 0.5).float()
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_loss = running_val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        
        # Imprimir métricas da época
        print(f"{epoch+1}\t{train_loss:.4f}\t\t{train_acc:.2f}%\t\t{val_loss:.4f}\t\t{val_acc:.2f}%")
        
        # Atualiza e salva modelo se houver melhoria na acurácia de validação
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }
            torch.save(checkpoint, 'models/best_model.pth')
            print(f"Best model saved with validation accuracy: {val_acc:.2f}%")
    
    print("\nTraining completed!")
    print(f"Best validation accuracy achieved: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()