from pathlib import Path
import argparse
import torch
import numpy as np
import nibabel as nib
from functools import partial

from monai.transforms import (
    Activations,
    LoadImaged,
    Compose,
    EnsureChannelFirstd,
    MapTransform,
    NormalizeIntensityd,
    ToTensord
)
from monai.data import DataLoader, CacheDataset, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR

parser = argparse.ArgumentParser(description='Swin UNETR segmentation inference')
parser.add_argument('--data_dir', default='/dataset/dataset0/', type=str, help='dataset directory')
parser.add_argument('--output_dir', default='test1', type=str, help='experiment output dir name')
parser.add_argument('--pretrained_model_name', default='model.pt', type=str, help='pretrained model name')
parser.add_argument('--feature_size', default=48, type=int, help='feature size')
parser.add_argument('--infer_overlap', default=0.5, type=float, help='sliding window inference overlap')
parser.add_argument('--in_channels', default=4, type=int, help='number of input channels')
parser.add_argument('--out_channels', default=3, type=int, help='number of output channels')
parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
parser.add_argument('--posrate', default=1.0, type=float, help='positive label rate')
parser.add_argument('--negrate', default=1.0, type=float, help='negative label rate')
parser.add_argument('--nsamples', default=1, type=int, help='number of croped samples')
parser.add_argument('--cacherate', default=1.0, type=float, help='cache data rate')
parser.add_argument('--workers', default=0, type=int, help='number of workers')
parser.add_argument('--batch_size', default=1, type=int, help='number of batch size')
parser.add_argument('--use_checkpoint', action='store_true', help='use gradient checkpointing to save memory')
parser.add_argument('--pretrained_model_path', default='./pretrained', type=str, help='pretrained checkpoint directory')
parser.add_argument('--pred_label', action='store_true', help='predict labels or regions')


class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on new brats 2023 classes:
    label 2 is the peritumoral edema
    label 3 is the GD-enhancing tumor
    label 1 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 1 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 1, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # label 3 is ET
            result.append(d[key] == 3)
            d[key] = torch.stack(result, axis=0).float()
        return d


def get_loader(args):
    """
    Load data sets for training, validation and testing from json files
    """
    data_root = Path(args.data_dir)
    channel_order= ['-t1n.nii.gz', '-t1c.nii.gz','-t2w.nii.gz','-t2f.nii.gz']
    img_paths = [f"{data_root.name}{c}" for c in channel_order]

    # val_data = json_data['validation']
    val_data = [{'image': img_paths}]
    # add data root to json file lists 
    for i in range(0, len(val_data)):
        # val_data[i]['label'] = str(data_root / val_data[i]['label'])
        for j in range(0, len(val_data[i]['image'])):
            val_data[i]['image'][j] = str(data_root / val_data[i]['image'][j])
        
    # define transforms
    val_transform = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ToTensord(keys=["image"]),
        ]
    )
    
    # define datasets
    val_ds = CacheDataset(
        data=val_data, 
        transform=val_transform, 
        cache_rate=args.cacherate, 
        num_workers=args.workers
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False
    )

    return val_loader
    

def main():
    args = parser.parse_args()
    output_directory = Path(args.output_dir)
    if not output_directory.exists():
        output_directory.mkdir(parents=True)

    val_loader = get_loader(args)    
    pretrained_pth = Path(args.pretrained_model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=args.use_checkpoint
    )

    model_dict = torch.load(pretrained_pth, map_location=device)
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)

    model_inferer_test = partial(
        sliding_window_inference,
        roi_size=[args.roi_x, args.roi_y, args.roi_z],
        sw_batch_size=1,
        predictor=model,
        overlap=args.infer_overlap,
    )

    with torch.no_grad():
        for _, batch in enumerate(val_loader):
            image = batch["image"].cuda()
            affine = batch['image_meta_dict']['original_affine'][0].numpy()
            filepath = Path(batch['image_meta_dict']['filename_or_obj'][0])
            names = filepath.name.split('-')
            img_name = f"{names[0]}-{names[1]}-{names[2]}-{names[3]}"
            output_pred =  model_inferer_test(image)
            print(f"Inference on case {img_name}")
            # print(f"Image shape: {image.shape}")
            # print(f"Prediction shape: {output_pred.shape}")
            post_trans = Activations(sigmoid=not args.pred_label, softmax=args.pred_label)
            prob = [post_trans(i) for i in decollate_batch(output_pred)]
            prob_np = prob[0].detach().cpu().numpy()
            # print(f"Probmap shape: {prob_np.shape}")
            np.savez(output_directory / f"{img_name}.npz", probabilities=prob_np)
            
            # if args.pred_label:
            #     seg_out = np.argmax(prob_np, axis=0)
            # else:
            #     seg = (prob_np > 0.5).astype(np.int8)
            #     seg_out = np.zeros_like(seg[0])
            #     seg_out[seg[1] == 1] = 2
            #     seg_out[seg[0] == 1] = 1
            #     seg_out[seg[2] == 1] = 3
            
            # nib.save(nib.Nifti1Image(seg_out.astype(np.int8), affine),
            #         output_directory / f"{img_name}.nii.gz")
                
            # print(f"Seg shape: {seg_out.shape}")
                 
        print("Finished inference!")


if __name__ == '__main__':
    main()
