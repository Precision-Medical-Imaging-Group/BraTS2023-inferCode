import subprocess

def check_path(path: str):
    '''
    Checks if the provided path points to a ZIP file.

    Parameters:
    path (str): Path to be checked.

    Raises:
    AssertionError: If the path does not have a ".zip" extension.
    '''
    
    assert path.endswith(".zip"), "Path must point to a ZIP file"

def install_nnunet_model_from_zip(path: str):
    '''
    Installs a pretrained nnU-Net (v2) model from a zipped file.

    Parameters:
    path (str): Path to the zipped model file.
    '''

    check_path(path)
    cmd = f"nnUNetv2_install_pretrained_model_from_zip {path}"
    subprocess.run(cmd, shell=True)  # Executes the command in the shell