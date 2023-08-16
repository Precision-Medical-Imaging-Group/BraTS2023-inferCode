import subprocess


def check_path(path: str):
    assert path.endswith(".zip"), "Path must point to a ZIP file"


def install_nnunet_model_from_zip(path: str):
    '''
    path: path where the zipped model is
    '''

    check_path(path)
    cmd = f"nnUNetv2_install_pretrained_model_from_zip {path}"
    subprocess.run(cmd, shell=True)  # Executes the command in the shell