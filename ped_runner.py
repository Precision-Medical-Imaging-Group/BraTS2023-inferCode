import tempfile
import os
from pathlib import Path
from nnunet.runner import run_infer_nnunet
from swinunetr.runner import run_infer_swinunter
from postproc.postprocess import remove_dir, postprocess_batch

CONSTANTS={
    'et_nnunet_model_path':'',
    'tcwt_nnunet_model_path':'',
    'swinunter_model_path':'',
    'et_ratio': 0.04,
    'ed_ratio':1.0,
    'remove_dir_factor':130,
}

def maybe_make_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def ped_ensembler(nnunet_et_npz_path, nnunet_tcwt_npz_path, swinunter_npz_path):
    ## ensemble code here
    ensembled = 
    return ensembled_path #niftiimage


def infer_single(input_path, out_dir):
    with tempfile.TemporaryDirectory() as tmpdirname:
        temp_dir = Path(tmpdirname)
        print(f'storing artifacts in tmp dir {temp_dir}')
        
        nnunet_et_npz_path = run_infer_nnunet(input_path, maybe_make_dir(temp_dir/ 'et'), CONSTANTS['et_nnunet_model_path'])
        nnunet_tcwt_npz_path = run_infer_nnunet(input_path, maybe_make_dir(temp_dir/ 'tcwt'), CONSTANTS['tcwt_nnunet_model_path'])
        swinunter_npz_path = run_infer_swinunter(input_path, maybe_make_dir(temp_dir/ 'swin'), CONSTANTS['swinunter_model_path'])

        ensemble_folder =  maybe_make_dir(temp_dir/ 'ensemble')
        ensembled_pred_nii_path = ped_ensembler(nnunet_et_npz_path, nnunet_tcwt_npz_path, swinunter_npz_path, ensemble_folder)

        label_to_optimize= 'et'
        pp_et_out = maybe_make_dir(temp_dir/ 'pp{label_to_optimize}')
        postprocess_batch(ensemble_folder, pp_et_out, label_to_optimize, ratio=CONSTANTS[f'{label_to_optimize}_ratio'], convert_to_brats_labels=False)
        
        label_to_optimize= 'ed'
        pp_ed_out = maybe_make_dir(temp_dir/ 'pp{label_to_optimize}')
        postprocess_batch(pp_et_out, pp_ed_out, label_to_optimize, ratio=CONSTANTS[f'{label_to_optimize}_ratio'], convert_to_brats_labels=False)
        
        remove_dir(pp_ed_out, maybe_make_dir(out_dir), CONSTANTS['remove_dir_factor'])