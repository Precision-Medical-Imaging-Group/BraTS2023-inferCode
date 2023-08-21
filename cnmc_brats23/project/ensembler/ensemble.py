import numpy as np
import nibabel as nib

def ped_ensembler(nnunet_et_npz_path_list, nnunet_tcwt_npz_path_list, swinunetr_npz_path_list, ensembled_path, input_img):

    if not ensembled_path.exists():
        ensembled_path.mkdir(parents=True)
    
    # ensemble SwinUNETR first
    case = swinunetr_npz_path_list[0].name.split('.npz')[0]
    print(f"Ensemble {case}")
    prob = np.load(swinunetr_npz_path_list[0])['probabilities']
    for i in range(1, len(swinunetr_npz_path_list)):
        prob += np.load(swinunetr_npz_path_list[i])['probabilities']
    prob_swin = prob / len(swinunetr_npz_path_list)
    print(f"Probabilities SwinUNETR: {prob_swin.shape}")
    
    # ensemble nnunet
    prob = np.load(nnunet_et_npz_path_list[0], allow_pickle=True)['probabilities']
    for i in range(1, len(nnunet_et_npz_path_list)):
        prob += np.load(nnunet_et_npz_path_list[i], allow_pickle=True)['probabilities']
    prob /= len(nnunet_et_npz_path_list)
    prob_et = prob[1]
    prob_et = np.swapaxes(prob_et, 0, 2)
    print(f"Probabilities nnunet ET: {prob_et.shape}")

    prob_tcwt = np.load(nnunet_tcwt_npz_path_list[0])['probabilities']
    for i in range(1, len(nnunet_tcwt_npz_path_list)):
        prob_tcwt += np.load(nnunet_tcwt_npz_path_list[1])['probabilities']
    prob_tcwt /= len(nnunet_tcwt_npz_path_list)
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

def men_ensembler(nnunet_npz_path_list, swinunetr_npz_path_list, ensembled_path, input_img):

    if not ensembled_path.exists():
        ensembled_path.mkdir(parents=True)
    
    # ensemble SwinUNETR first
    case = swinunetr_npz_path_list[0].name.split('.npz')[0]
    print(f"Ensemble {case}")
    prob = np.load(swinunetr_npz_path_list[0])['probabilities']
    for i in range(1, len(swinunetr_npz_path_list)):
        prob += np.load(swinunetr_npz_path_list[i])['probabilities']
    prob_swin = prob / len(swinunetr_npz_path_list)
    print(f"Probabilities SwinUNETR: {prob_swin.shape}")
    
    # ensemble nnunet
    prob = np.load(nnunet_npz_path_list[0], allow_pickle=True)['probabilities']
    for i in range(1, len(nnunet_npz_path_list)):
        prob += np.load(nnunet_npz_path_list[i], allow_pickle=True)['probabilities']
    prob /= len(nnunet_npz_path_list)
    prob_tc = prob[1]
    prob_tc = np.swapaxes(prob_tc, 0, 2)
    print(f"Probabilities nnunet TC: {prob_tc.shape}")
    
    prob_wt = prob[0]
    prob_wt = np.swapaxes(prob_wt, 0, 2)
    print(f"Probabilities nnunet WT: {prob_wt.shape}")

    prob_et = prob[2]
    prob_et = np.swapaxes(prob_et, 0, 2)
    print(f"Probabilities nnunet ET: {prob_et.shape}")
    
    prob_wt = (prob_wt + prob_swin[1]) * 0.5
    prob_tc = (prob_tc + prob_swin[0]) * 0.5
    prob_et = (prob_et + prob_swin[2]) * 0.5
    prob_out = np.zeros_like(prob_swin)
    prob_out[0] = prob_tc
    prob_out[1] = prob_wt
    prob_out[2] = prob_et

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

def met_ensembler(nnunet_npz_path_list, ensembled_path, input_img):
    if not ensembled_path.exists():
        ensembled_path.mkdir(parents=True)
    
    # ensemble nnunet
    case = nnunet_npz_path_list[0].name.split('.npz')[0]
    print(f"Ensemble {case}")
    prob = np.load(nnunet_npz_path_list[0], allow_pickle=True)['probabilities']
    for i in range(1, len(nnunet_npz_path_list)):
        prob += np.load(nnunet_npz_path_list[i], allow_pickle=True)['probabilities']
    prob /= len(nnunet_npz_path_list)

    prob_tc = prob[1]
    prob_tc = np.swapaxes(prob_tc, 0, 2)
    print(f"Probabilities nnunet TC: {prob_tc.shape}")
    
    prob_wt = prob[0]
    prob_wt = np.swapaxes(prob_wt, 0, 2)
    print(f"Probabilities nnunet WT: {prob_wt.shape}")

    prob_et = prob[2]
    prob_et = np.swapaxes(prob_et, 0, 2)
    print(f"Probabilities nnunet ET: {prob_et.shape}")
    
    prob_out = np.stack([prob_tc, prob_wt, prob_et], axis=0)
    
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