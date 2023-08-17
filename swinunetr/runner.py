import subprocess
from pathlib import Path


def run_infer_swinunetr(input_path: Path, output_folder: Path, model_folder_pth: Path, 
                        challenge_name: str, folds=[0,1,2,3,4])->list:
    """runner that helps run swinunter based on a model path and 
    returns path to the probability npz

    Args:
        input_path (Path): path to the folder containing the 4 input files 
        output_path (Path): path to folder to store teh npz file
        model_folder_pth (Path): path where model weights are stored

    Returns:
        path: path to the npz file
    """
    pretrained_name = 'best_checkpoint.pt'
    npz_folder_list = []
    for fold in folds:
        pretrained_path = model_folder_pth / f"fold_{fold}" 
        output_dir = output_folder / f"{challenge_name}_swin_region_f{fold}"
        npz_folder_list.append(output_dir)
        cmd = 'python inference.py'
        cmd = ' '.join((cmd, f'--data_dir={str(input_path)}'))
        cmd = ' '.join((cmd, f'--output_dir={str(output_dir)}'))
        cmd = ' '.join((cmd, '--roi_x=96'))
        cmd = ' '.join((cmd, '--roi_y=96'))
        cmd = ' '.join((cmd, '--roi_z=96'))
        cmd = ' '.join((cmd, '--in_channels=4'))
        cmd = ' '.join((cmd, '--out_channels=3'))
        cmd = ' '.join((cmd, '--use_checkpoint'))
        cmd = ' '.join((cmd, '--feature_size=48'))
        cmd = ' '.join((cmd, '--infer_overlap=0.5'))
        cmd = ' '.join((cmd, '--cacherate=1.0'))
        cmd = ' '.join((cmd, '--workers=0'))
        cmd = ' '.join((cmd, f'--pretrained_model_name={pretrained_name}'))
        cmd = ' '.join((cmd, f'--pretrained_dir={str(pretrained_path)}'))
        # cmd = ' '.join((cmd, '--pred_label'))
        print(cmd)
        subprocess.run(cmd, shell=True)  # Executes the command in the shell
    
    npz_path_list = [f / f"{input_path.name}.npz" for f in npz_folder_list]
    return npz_path_list