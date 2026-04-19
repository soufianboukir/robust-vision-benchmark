import os

class Config:

    batch_size = 128
    learning_rate = 1e-3
    epochs = 15

    device = "cuda"

    model_type = "mlp3"

    data_path = "./data"
    save_dir = os.path.join(os.getcwd(), "saved_models")