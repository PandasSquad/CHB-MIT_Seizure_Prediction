"""Train the model."""
import mlflow
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import random_split
from pytorch_lightning.loggers import MLFlowLogger
from dataset import CustomDataset
from models import MLP
import json

mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="../mlruns")

# Define los parámetros de tu experimento
params = {
    "batch_size": 64,
    "lr": 0.001,
    "epochs": 30,
}

# Carga los datos
folder_path = "data/windows_per_csv"
dataset = CustomDataset(folder_path)

# Divide los datos
train_len = int(len(dataset) * 0.7)
val_len = (len(dataset) - train_len) // 2
test_len = len(dataset) - train_len - val_len
train_data, val_data, test_data = random_split(dataset, [train_len, val_len, test_len])

# Crea la instancia del modelo
model = MLP(train_data=train_data, val_data=val_data, test_data=test_data)

# Callback para que no se estanque el entrenamiento
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.00,
    patience=5,
    verbose=False,
    mode="min",
)

trainer = Trainer(
    max_epochs=params["epochs"],
    callbacks=[early_stop_callback],
    logger=mlf_logger,
)


# Comienza el experimento de MLFlow
with mlflow.start_run():
    # Registra los parámetros
    mlflow.log_params(params)

    # Entrena el modelo
    trainer.fit(
        model,
        train_dataloaders=model.train_dataloader(),
        val_dataloaders=model.val_dataloader(),
    )

    # Registra el modelo en MLFlow
    mlflow.pytorch.log_model(model, "model_1_MLP")

    # Evalúa el modelo
    trainer.test()

    # Registra las métricas de prueba en MLFlow
    for key, value in trainer.logged_metrics.items():
        mlflow.log_metric(key, value)

    # Save logged_metrics to a json file
    with open("metrics.json", "w") as f:
        json.dump(trainer.logged_metrics, f)
