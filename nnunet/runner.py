import subprocess
from tqdm import tqdm
import os

def maybe_make_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def set_env_paths(input_path, path):
    #raw_path = maybe_make_dir(path / 'nnUNet_raw')
    preprocessed = maybe_make_dir(input_path / 'nnUNet_preprocessed')
    #results =  maybe_make_dir(path / 'nnUNet_results')
    return f"nnUNet_raw='{input_path}' nnUNet_preprocessed='{preprocessed}' nnUNet_results='./'"

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
    elif(challenge_name=="BraTS2023_PED_ET"):
        dataset_name = "Dataset008_BraTS2023_PED_ONLY_ET"
    else:
        raise Exception("Challenge name not compatible.")
    
    return dataset_name

def run_infer_nnunet(input_folder: str, output_folder: str,  challenge_name: str, name,  folds=[0,1,2,3,4], save_npz=True, ensemble=True)->list:
    """_summary_

    Args:
        input_folder (str): input folder
        output_folder (str): folder where output is to be stored
        challenge_name (str): task name for which inference is done
        name (str): file name
        folds (list, optional): describe which folds to run. Defaults to [0,1,2,3,4].
        save_npz (bool, optional): whether to save the probabilities. Defaults to True.

    Returns:
        list: a list of the npz files from each fold
    """
    
    # Check challenge name and get dataset name
    dataset_name = get_dataset_name(challenge_name)
    env_set = set_env_paths(input_folder, output_folder)

    # Variables
    trainer_name = "nnUNetTrainer_100epochs"
    configuration_name = "3d_fullres"

    if ensemble:
        output_folder_fold = os.path.join(output_folder,"ens")
        print(f"Running nnUnet inference with all folds (ensemble)..")
        cmd = f"{env_set} nnUNetv2_predict -i '{input_folder}' -o '{output_folder_fold}' -d '{dataset_name}' -c '{configuration_name}' -tr '{trainer_name}'"
        if(save_npz):
            cmd+=" --save_probabilities"
        subprocess.run(cmd, shell=True)

        return [os.path.join(output_folder_fold, name+'.npz')]
    else:
        npz_path_list = [] 
        for fold in tqdm(folds):
            output_folder_fold = os.path.join(output_folder, f"fold_{fold}")
            print(f"Running nnU-Net inference for fold {fold}")
            cmd = f"{env_set} nnUNetv2_predict -i '{input_folder}' -o '{output_folder_fold}' -d '{dataset_name}' -c '{configuration_name}' -tr '{trainer_name}' -f '{fold}'"
            if(save_npz):
                cmd+=" --save_probabilities"
            subprocess.run(cmd, shell=True)  # Executes the command in the shell
            npz_path_list.append(os.path.join(output_folder_fold, name+'.npz'))

        return npz_path_list