import tempfile
import os
import shutil
from pathlib import Path
import time

from nnunet.install_model import install_model_from_zip
from ensembler.ensemble import men_met_ensembler
from nnunet.runner import run_infer_nnunet
from swinunetr.runner import run_infer_swinunetr
from postproc.postprocess import remove_disconnected_from_dir

NAME_MAPPER ={
    '-t1n.nii.gz': '_0000.nii.gz',
    '-t1c.nii.gz': '_0001.nii.gz',
    '-t2w.nii.gz': '_0002.nii.gz',
    '-t2f.nii.gz': '_0003.nii.gz'
}
CONSTANTS={
    'nnunet_model_path':'./weights/BraTS2023_MEN_nnunetv2_model.zip',
    'swinunetr_model_path':'./weights/BraTS2023-MEN-Model-Swin.zip',
    'swinunetr_pt_path':'./BraTS2023-MEN-Model-Swin',
    'remove_disconnected_factor': 110,
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
        
        nnunet_npz_path_list = run_infer_nnunet(input_folder_raw/ name, maybe_make_dir(temp_dir/ 'nnunet'), 'BraTS2023_MEN', name)
        swinunetr_npz_path_list = run_infer_swinunetr(Path(input_path), maybe_make_dir(temp_dir/ 'swin'), 'men', Path(CONSTANTS['swinunetr_pt_path']))
        
        ensemble_folder =  maybe_make_dir(temp_dir/ 'ensemble')
        ensembled_pred_nii_path = men_met_ensembler(nnunet_npz_path_list, swinunetr_npz_path_list, ensemble_folder, one_image)

        remove_disconnected_from_dir(ensemble_folder, maybe_make_dir(out_dir), CONSTANTS['remove_disconnected_factor'])

def setup_model_weights():
    install_model_from_zip(CONSTANTS['nnunet_model_path'])
    install_model_from_zip(CONSTANTS['swinunetr_model_path'])

def batch_processor(input_folder, output_folder):
    for input_path in Path(input_folder).iterdir():
        infer_single(input_path, output_folder)


if __name__ == "__main__":
    setup_model_weights()
    start_time = time.time()
    input_path = '/media/abhijeet/Seagate Portable Drive1/Brats23/BRATS Pediatric Dataset/ASNR-MICCAI-BraTS2023-MEN-Challenge-ValidationData/'
    output_folder = './output_for_comp/'
    batch_processor(input_path, output_folder)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {(elapsed_time/60):.1f} minutes")
    print(f"Time taken per sample: {(elapsed_time/60/45):.1f} minutes")
