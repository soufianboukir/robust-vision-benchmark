import os

class Config:

    batch_size = 64
    learning_rate = 8e-4
    epochs = 20

    device = "cuda"

    model_type = "resnet50"

    data_path = "./data"
    save_dir = os.path.join(os.getcwd(), "saved_models")