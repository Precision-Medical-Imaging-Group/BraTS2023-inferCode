import subprocess
from tqdm import tqdm
import os

def maybe_make_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def set_env_paths(path):
    raw_path = maybe_make_dir(path / 'nnUNet_raw')
    preprocessed = maybe_make_dir(path / 'nnUNet_preprocessed')
    results =  maybe_make_dir(path / 'nnUNet_results')
    return f"nnUNet_raw='{raw_path}' nnUNet_preprocessed='{preprocessed}' nnUNet_results='{results}'"

def get_dataset_name(challenge_name: str):
    '''
    Returns the corresponding dataset name based on the provided challenge name.

    Parameters:
    challenge_name (str): Name of the challenge.

    Returns:
    str: Corresponding dataset name.
    
    Raises:
    Exception: If the challenge name is not compatible.
    '''
    
    if(challenge_name=="BraTS2023_MEN"):
        dataset_name = "Dataset004_BraTS2023_MEN"
    elif(challenge_name=="BraTS2023_MET"):
        dataset_name = "Dataset005_BraTS2023_MET"
    elif(challenge_name=="BraTS2023_PED"):
        dataset_name = "Dataset006_BraTS2023_PED"
    else:
        raise Exception("Challenge name not compatible.")
    
    return dataset_name

def run_infer_nnunet(input_folder: str, output_folder: str, challenge_name: str, folds=[0,1,2,3,4], save_npz=True)->list:
    '''
    Runs nnU-Net inference for a given set of parameters.

    Parameters:
    input_folder (str): Input folder containing NIfTI files to be predicted.
    output_folder (str): Output folder where predictions will be saved.
    challenge_name (str): Name of the challenge.
    folds (list, optional): List of folds for which to run inference. Defaults to [0,1,2,3,4].
    save_npz (bool, optional): Whether to save prediction probabilities as npz files (apart from NIfTI). Defaults to True.
    '''
    
    # Check challenge name and get dataset name
    dataset_name = get_dataset_name(challenge_name)
    env_set = set_env_paths(output_folder)
    
    # Variables
    trainer_name = "nnUNetTrainer_100epochs"
    configuration_name = "3d_fullres"
    
    # Commands
    for fold in tqdm(folds):
        print(f"Running nnU-Net inference for fold {fold}")
        cmd = f"{env_set} nnUNetv2_predict -i {input_folder} -o {output_folder} -d {dataset_name} -c {configuration_name} -tr {trainer_name} -f {fold}"
        if(save_npz):
            cmd+=" --save_probabilities"
        subprocess.run(cmd, shell=True)  # Executes the command in the shell

    # get the final npz(single file)
    # TODO: can you please return a list of npz_path
    # TODO: we dont care about separate folders as long as you return the list as below
    # remember try not to generate more file than the 5 unless you really have to 
    npz_path_list = [npz_path_0, npz_path_1, npz_path_2, npz_path_3, npz_path_4]
    return npz_path_list