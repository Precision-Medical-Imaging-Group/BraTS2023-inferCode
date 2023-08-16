

def run_infer_swinunetr(input_path, output_folder, model_folder_pth):
    """runner that helps run swinunter based on a model path and 
    returns path to the probability npz

    Args:
        input_path (path): path to the folder containing the 4 input files 
        output_path (path): path to folder to store teh npz file
        model_folder_pth (path): path where model weights are stored

    Returns:
        path: path to the npz file
    """
    
    return path_npz