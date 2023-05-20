"""Train script."""
import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow
from dataset import CustomDataset
from models import MLP

# autolog
mlflow.pytorch.autolog()


def train() -> None:
    """Train the model."""
    folder_path = "data/windows_per_csv"
    print("Loading dataset...")
    dataset = CustomDataset(folder_path)
    print("Dataset loaded.")
    batch_size = 32
    print("Creating dataloader...")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("Dataloader created.")

    model = MLP()
    criterion = nn.CrossEntropyLoss()  # Función de pérdida
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizador

    # Entrenamiento del modelo
    num_epochs = 10

    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs = inputs.float()  # Convert inputs to float
            labels = labels.long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # type: ignore
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            accuracy = (outputs.argmax(dim=1) == labels).float().mean()

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}")
        mlflow.log_metric("loss", epoch_loss)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.pytorch.log_model(model, "models")

    print("Entrenamiento finalizado.")


if __name__ == "__main__":
    # check if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    # mlflow settings
    # mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("MLP EEG")

    with mlflow.start_run():
        train()
