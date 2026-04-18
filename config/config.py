class Config:

    batch_size = 128
    learning_rate = 1e-3
    epochs = 15

    device = "cuda"

    model_type = "cnn3"

    data_path = "./data"
    save_dir = "../../saved_models/mlp3.pth"