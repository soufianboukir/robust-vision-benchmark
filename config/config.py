import os

class Config:

    batch_size = 64
    learning_rate = 1e-3
    epochs = 30

    device = "cuda"

    model_type = "resnet18"

    data_path = "./data"
    save_dir = os.path.join(os.getcwd(), "saved_models")