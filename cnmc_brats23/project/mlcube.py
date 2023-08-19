"""MLCube handler file"""
import os 
os.environ["MKL_SERVICE_FORCE_INTEL"] = '1'  # see issue #152
import typer
import tempfile
import os
import shutil
from pathlib import Path
import time

from nnunet.install_model import install_model_from_zip
from ensembler.ensemble import ped_ensembler
from nnunet.runner import run_infer_nnunet
from swinunetr.runner import run_infer_swinunetr
from postproc.postprocess import remove_dir, postprocess_batch

NAME_MAPPER ={
    '-t1n.nii.gz': '_0000.nii.gz',
    '-t1c.nii.gz': '_0001.nii.gz',
    '-t2w.nii.gz': '_0002.nii.gz',
    '-t2f.nii.gz': '_0003.nii.gz'
}
CONSTANTS={
    'et_nnunet_model_path':'./weights/BraTS2023_PED_ONLY_ET_nnunetv2_model.zip',
    'tcwt_nnunet_model_path':'./weights/BraTS2023_PED_nnunetv2_model.zip',
    'swinunter_model_path':'./weights/BraTS2023-Swin-Weights.zip',
    'swinunter_pt_path':'./BraTS2023-Swin-Weights',
    'et_ratio': 0.04,
    'ed_ratio':1.0,
    'remove_dir_factor':130,
}

def maybe_make_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

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
        name = input_path.name
        print(name)
        shutil.copytree(input_path, input_folder_raw / name)
        # only for the test
        #os.remove(input_folder_raw / name/ f'{name}-seg.nii.gz')
        for key, val in NAME_MAPPER.items():
            os.rename(input_folder_raw / name/ f'{name}{key}', input_folder_raw / name/ f'{name}{val}')
            one_image = input_folder_raw / name/ f'{name}{val}'
        
        nnunet_et_npz_path_list = run_infer_nnunet(input_folder_raw/ name, maybe_make_dir(temp_dir/ 'et'), 'BraTS2023_PED_ET', name)
        nnunet_tcwt_npz_path_list = run_infer_nnunet(input_folder_raw/ name, maybe_make_dir(temp_dir/ 'tcwt'), 'BraTS2023_PED', name)
        swinunter_npz_path_list = run_infer_swinunetr(Path(input_path), maybe_make_dir(temp_dir/ 'swin'), 'ped', Path(CONSTANTS['swinunter_pt_path']))
        
        ensemble_folder =  maybe_make_dir(temp_dir/ 'ensemble')
        ensembled_pred_nii_path = ped_ensembler(nnunet_et_npz_path_list, nnunet_tcwt_npz_path_list, swinunter_npz_path_list, ensemble_folder, one_image)
        
        pp_folder =  maybe_make_dir(temp_dir/ 'cc')
        remove_dir(ensemble_folder, pp_folder, 50)

        label_to_optimize= 'et'
        pp_et_out = maybe_make_dir(temp_dir/ 'pp{label_to_optimize}')
        postprocess_batch(pp_folder, pp_et_out, label_to_optimize, ratio=CONSTANTS[f'{label_to_optimize}_ratio'], convert_to_brats_labels=False)
        
        label_to_optimize= 'ed'
        #pp_ed_out = maybe_make_dir(temp_dir/ 'pp{label_to_optimize}')
        postprocess_batch(pp_et_out,  maybe_make_dir(out_dir), label_to_optimize, ratio=CONSTANTS[f'{label_to_optimize}_ratio'], convert_to_brats_labels=False)

        # label_to_optimize= 'et'
        # pp_et_out = maybe_make_dir(temp_dir/ 'pp{label_to_optimize}')
        # postprocess_batch(ensemble_folder, pp_et_out, label_to_optimize, ratio=CONSTANTS[f'{label_to_optimize}_ratio'], convert_to_brats_labels=False)
        
        # label_to_optimize= 'ed'
        # pp_ed_out = maybe_make_dir(temp_dir/ 'pp{label_to_optimize}')
        # postprocess_batch(pp_et_out, pp_ed_out, label_to_optimize, ratio=CONSTANTS[f'{label_to_optimize}_ratio'], convert_to_brats_labels=False)
        
        # remove_dir(pp_ed_out, maybe_make_dir(out_dir), CONSTANTS['remove_dir_factor'])


def setup_model_weights():
    install_model_from_zip(CONSTANTS['et_nnunet_model_path'])
    install_model_from_zip(CONSTANTS['tcwt_nnunet_model_path'])
    install_model_from_zip(CONSTANTS['swinunter_model_path'])

def batch_processor(input_folder, output_folder):
    for input_path in Path(input_folder).iterdir():
        infer_single(input_path, output_folder)

app = typer.Typer()


@app.command("infer")
def infer(
    data_path: str = typer.Option(..., "--data_path"),
    output_path: str = typer.Option(..., "--output_path"),

):
    setup_model_weights()
    batch_processor(data_path, output_path)
    return output_path



@app.command("hotfix")
def hotfix():
    # NOOP command for typer to behave correctly. DO NOT REMOVE OR MODIFY
    pass


if __name__ == "__main__":
    app()
