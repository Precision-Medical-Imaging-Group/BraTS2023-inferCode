import tempfile
import os
import shutil
from pathlib import Path
import numpy as np
import nibabel as nib
from nnunet.install_model import install_nnunet_model_from_zip
from nnunet.runner import run_infer_nnunet
from swinunetr.runner import run_infer_swinunetr
from postproc.postprocess import remove_dir, postprocess_batch

CONSTANTS={
    'et_nnunet_model_path':'./weights/BraTS2023_PED_ONLY_ET_nnunetv2_model.zip',
    'tcwt_nnunet_model_path':'./weights/BraTS2023_PED_nnunetv2_model.zip',
    'swinunter_model_path':'',
    'et_ratio': 0.04,
    'ed_ratio':1.0,
    'remove_dir_factor':130,
}

def maybe_make_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def ped_ensembler(nnunet_et_npz_path_list, nnunet_tcwt_npz_path_list, swinunter_npz_path_list, ensembled_path, input_img):
    assert len(nnunet_et_npz_path_list) == 5
    assert len(nnunet_tcwt_npz_path_list) == 5
    assert len(swinunter_npz_path_list) == 5

    if not ensembled_path.exists():
        ensembled_path.mkdir(parents=True)
    
    # ensemble SwinUNETR first
    case = swinunter_npz_path_list[0].name.split('.npz')[0]
    print(f"Ensemble {case}")
    prob = np.load(swinunter_npz_path_list[0])['probabilities']
    for i in range(1, 5):
        prob += np.load(swinunter_npz_path_list[i])['probabilities']
    prob_swin = prob / 5
    print(f"Probabilities SwinUNETR: {prob_swin.shape}")
    
    # ensemble nnunet
    prob = np.load(nnunet_et_npz_path_list[0])['probabilities']
    for i in range(1, 5):
        prob += np.load(nnunet_et_npz_path_list[i])['probabilities']
    prob /= 5
    prob_et = prob[1]
    prob_et = np.swapaxes(prob_et, 0, 2)
    print(f"Probabilities nnunet ET: {prob_et.shape}")

    prob_tcwt = np.load(nnunet_tcwt_npz_path_list[0])['probabilities']
    for i in range(1, 5):
        prob_tcwt += np.load(nnunet_tcwt_npz_path_list[1])['probabilities']
    prob_tcwt /= 5 
    prob_tc = prob_tcwt[1]
    prob_tc = np.swapaxes(prob_tc, 0, 2)
    print(f"Probabilities nnunet TC: {prob_tc.shape}")
    prob_wt = prob_tcwt[0]
    prob_wt = np.swapaxes(prob_wt, 0, 2)
    print(f"Probabilities nnunet WT: {prob_wt.shape}")
        
    prob_wt = (prob_wt + prob_swin[1]) * 0.5
    prob_tc = (prob_tc + prob_swin[0]) * 0.5
    prob_et = (prob_et + prob_swin[2]) * 0.5
    prob_out = np.zeros_like(prob_swin)
    prob_out[0] = prob_tc
    prob_out[1] = prob_wt
    prob_out[2] = prob_et
    # np.savez(output_prob / f"{case}.npz", probabilities=prob_out)

    # save seg
    seg = (prob_out > 0.5).astype(np.int8)
    seg_out = np.zeros_like(seg[0])
    seg_out[seg[1] == 1] = 2
    seg_out[seg[0] == 1] = 1
    seg_out[seg[2] == 1] = 3
    print(f"Seg: {seg.shape}")
    img = nib.load(input_img)
    nib.save(nib.Nifti1Image(seg_out.astype(np.int8), img.affine), ensembled_path / f"{case}.nii.gz")
    return ensembled_path / f"{case}.nii.gz"


def infer_single(input_path, out_dir):
    """do inference on a single folder

    Args:
        input_path (path): input folder, where the 4 nii.gz are stored
        out_dir (path): out folder, where the seg.nii.gz is to be stored
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_dir = Path(tmpdirname)
        print(f'storing artifacts in tmp dir {temp_dir}')
        input_folder_raw = maybe_make_dir(temp_dir/ 'inp')
        name = input_path.split('/')[-1]
        shutil.copytree(input_path, input_folder_raw / name)
        # only for the test
        os.remove(input_folder_raw / name/ f'{name}-seg.nii.gz')
        os.rename(input_folder_raw / name/ f'{name}-t1n.nii.gz', input_folder_raw / name/ f'{name}_0000.nii.gz')
        os.rename(input_folder_raw / name/ f'{name}-t1c.nii.gz', input_folder_raw / name/ f'{name}_0001.nii.gz')
        os.rename(input_folder_raw / name/ f'{name}-t2w.nii.gz', input_folder_raw / name/ f'{name}_0002.nii.gz')
        os.rename(input_folder_raw / name/ f'{name}-t2f.nii.gz', input_folder_raw / name/ f'{name}_0003.nii.gz')

        nnunet_et_npz_path_list = run_infer_nnunet(input_folder_raw/ name, maybe_make_dir(temp_dir/ 'et'), 'BraTS2023_PED_ET', name)
        nnunet_tcwt_npz_path_list = run_infer_nnunet(input_folder_raw/ name, maybe_make_dir(temp_dir/ 'tcwt'), 'BraTS2023_PED', name)
        print(nnunet_et_npz_path_list)
        print(nnunet_tcwt_npz_path_list)
        print('it works untill here...')
        # swinunter_npz_path_list = run_infer_swinunter(input_folder_raw/ name, maybe_make_dir(temp_dir/ 'swin'), CONSTANTS['swinunter_model_path'])

        # ensemble_folder =  maybe_make_dir(temp_dir/ 'ensemble')
        # ensembled_pred_nii_path = ped_ensembler(nnunet_et_npz_path_list, nnunet_tcwt_npz_path_list, swinunter_npz_path_list, ensemble_folder)

        # label_to_optimize= 'et'
        # pp_et_out = maybe_make_dir(temp_dir/ 'pp{label_to_optimize}')
        # postprocess_batch(ensemble_folder, pp_et_out, label_to_optimize, ratio=CONSTANTS[f'{label_to_optimize}_ratio'], convert_to_brats_labels=False)
        
        # label_to_optimize= 'ed'
        # pp_ed_out = maybe_make_dir(temp_dir/ 'pp{label_to_optimize}')
        # postprocess_batch(pp_et_out, pp_ed_out, label_to_optimize, ratio=CONSTANTS[f'{label_to_optimize}_ratio'], convert_to_brats_labels=False)
        
        # remove_dir(pp_ed_out, maybe_make_dir(out_dir), CONSTANTS['remove_dir_factor'])


if __name__ == "__main__":
    install_nnunet_model_from_zip(CONSTANTS['et_nnunet_model_path'])
    install_nnunet_model_from_zip(CONSTANTS['tcwt_nnunet_model_path'])
    input_path = 'path_to_data/ASNR-MICCAI-BraTS2023-PED-Challenge-TrainingData/BraTS-PED-00002-000'
    infer_single(input_path, './')
