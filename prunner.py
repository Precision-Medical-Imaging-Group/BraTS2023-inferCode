import torch
fold = 0
for fold in [0, 1, 2, 3, 4]:
    x = torch.load(f'/home/abhijeet/BraTS2023-MEN-Model-swin/fold{fold}/best_checkpoint.pt')['model']
    torch.save(x, f'/home/abhijeet/BraTS2023-MEN-Model-swin/ckpt_men_f{fold}.pt')