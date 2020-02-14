import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)