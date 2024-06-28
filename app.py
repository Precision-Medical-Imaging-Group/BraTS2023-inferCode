import logging
logger = logging.getLogger(__file__)
import gradio as gr
import numpy as np
import SimpleITK as sitk
import PIL
from pathlib import Path
from app_assets import logo
import glob


import subprocess

mydict = {}

def run_inference(image_path, image_path2, image_path3, image_path4):
    input_path = image_path.parent / image_path.name[:-11]
    cmd = f"mkdir {input_path}; mv {image_path} {input_path}; mv {image_path2} {input_path}; mv {image_path3} {input_path}; mv {image_path4} {input_path}"
    print(cmd)
    subprocess.run(cmd, shell=True)
    cmd = f"mkdir {'./outs'}; cd cnmc_brats23/mlcube; mlcube run --gpus device=1 --task infer data_path={input_path.parent} output_path={'../outs'}"
    print(cmd)
    subprocess.run(cmd, shell=True)

    return input_path, f'./cnmc_brats23/mlcube/outs/{image_path.name[:-11]}.nii.gz'

def get_img_mask(image_path, mask_path):
    img_obj = sitk.ReadImage(image_path)
    mask_obj = sitk.ReadImage(mask_path)
    img = sitk.GetArrayFromImage(img_obj)
    mask = sitk.GetArrayFromImage(mask_obj)
    minval, maxval = np.min(img), np.max(img)
    img = ((img-minval)/(maxval-minval)).clip(0,1)*255

    return img, img_obj, mask

def main_func(file_obj):
    print(file_obj)
    global mydict
    image_path = file_obj[0]
    image_path, mask_path = run_inference(Path(file_obj[0]), Path(file_obj[1]),Path(file_obj[2]), Path(file_obj[3]))
    image_path = glob.glob(str(image_path / '*.nii.gz'))
    mydict['img_path'] = image_path
    mydict['mask_path'] = mask_path
    img, img_obj, mask = get_img_mask(image_path[0], mask_path)
    
    mydict['img'] = img.astype(np.uint8)
    mydict['mask'] = mask.astype(np.uint8)

    print(img_obj.GetSpacing())
    spacing_tuple = img_obj.GetSpacing()
    multiplier_ml = 0.001 * spacing_tuple[0] * spacing_tuple[1] * spacing_tuple[2]
    unique, frequency = np.unique(mask, return_counts = True)
    for i, lbl in enumerate(unique):
        mydict[f'vol_lbl{lbl}'] = multiplier_ml * frequency[i]


    return mask_path, f"Segmentation done! Total volume segmented {mydict.get('vol_lbl3', 0) + mydict.get('vol_lbl2', 0) + mydict.get('vol_lbl1', 0):.3f} ml; EDEMA {mydict.get('vol_lbl2', 0):.3f} ml; NECROSIS {mydict.get('vol_lbl3', 0):.3f} ml; ENHANCING TUMOR {mydict.get('vol_lbl1', 0):.3f} ml"

def main_func_example(file_obj):
    print(file_obj)
    mask_path, output_text = main_func(file_obj)
    x, state = 10, 10
    im, another_output_text = render(x, state)
    return mask_path, im, another_output_text

def render(file_to_render, x, state, view):
    global mydict
    suffix ={'T2 Flair':'t2f','native T1':'t1n', 'post-contrast T1-weighted':'t1c', 'T2 weighted': 't2w'}
    print(suffix[file_to_render])
    if len(mydict)>=4:
        print(mydict['img'])
        get_file = [file for file in mydict['img_path'] if suffix[file_to_render] in file][0]
        mydict['img'], _, _ =get_img_mask(get_file, get_file) 
        if x < 0:
            x = 0
        if view == 'axial':
            if x > mydict['img'].shape[0]-1:
                x = mydict['img'].shape[0]-1
            
            img = mydict['img'][x,:,:]
            mask = mydict['mask'][x,:,:]
        if view == 'coronal':
            if x > mydict['img'].shape[1]-1:
                x = mydict['img'].shape[1]-1
            
            img = mydict['img'][:,x,:]
            mask = mydict['mask'][:,x,:]
        if view == 'saggital':
            if x > mydict['img'].shape[2]-1:
                x = mydict['img'].shape[2]-1
            
            img = mydict['img'][:,:,x]
            mask = mydict['mask'][:,:,x]
        img = np.flipud(img)
        mask = np.flipud(mask)
        im = PIL.Image.fromarray(img)
        value = (im,[(mask==1,f"enhancing tumor: {mydict.get('vol_lbl1', 0):.3f} ml"),(mask==2,f"edema: {mydict.get('vol_lbl2', 0):.3f} ml"),(mask==3,f"necrosis: {mydict.get('vol_lbl3', 0):.3f} ml")])
        zmin, zmax = 0,mydict['img'].shape[0]-1
    else:
        im = np.zeros(10,10)
        zmin, zmax = None, None
        value = (im,[])

    return value#,f'z-value: {x}, (zmin: {zmin}, zmax: {zmax})'

def render_axial(file_to_render, x, state):
    return render(file_to_render, x, state, 'axial')
def render_coronal(file_to_render, x, state):
    return render(file_to_render, x, state, 'coronal')
def render_saggital(file_to_render, x, state):
    return render(file_to_render, x, state, 'saggital')


with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown(
        """
        # CNMC PMI Pediatric Brain Tumor Segmentation
        """)
    gr.HTML(value=f"<p style='margin-top: 1rem, margin-bottom: 1rem'> <img src='{logo.logo}' alt='Childrens National Logo' style='display: inline-block'/></p>")

    with gr.Row():
            image_folder= gr.Files(file_count="multiple", label="upload t1n, t1c, t2w and t2f files here:", file_types=["nii.gz"])
            print("image_folder", image_folder)
    with gr.Row():
        with gr.Column():
             gr.Button("", render=False)
        with gr.Column():
            btn = gr.Button("start segmentation")
        with gr.Column():
             gr.Button("", render=False)
    with gr.Column():
        out_text = gr.Textbox(label='Status', placeholder="Volumetrics will be updated here.")

    with gr.Row():
        with gr.Column():
             gr.Button("", render=False)
        with gr.Column():
            file_to_render = gr.Dropdown(['T2 Flair','native T1', 'post-contrast T1-weighted', 'T2 weighted'], label='choose file to overlay')
        with gr.Column():
             gr.Button("", render=False)

    with gr.Row():
        height = "20vw"
        myimage_axial = gr.AnnotatedImage(label="axial view", height=height)
        myimage_coronal = gr.AnnotatedImage(label="coronal view",height=height)
        myimage_saggital = gr.AnnotatedImage(label="saggital view",height=height)
    with gr.Row():
        slider_axial = gr.Slider(0, 155, step=1) # max val needs to be updated by user.
        state_axial = gr.State(value=0)
        slider_coronal = gr.Slider(0, 155, step=1) # max val needs to be updated by user.
        state_coronal = gr.State(value=0)
        slider_saggital = gr.Slider(0, 155, step=1) # max val needs to be updated by user.
        state_saggital = gr.State(value=0)

    with gr.Row():
        mask_file = gr.File(label="download annotation", height="vw" )

    example_1 = [[
        "/home/pmilab/Xinyang/data/BraTS2023/ASNR-MICCAI-BraTS2023-PED-Challenge-ValidationData/BraTS-PED-00210-000/BraTS-PED-00210-000-t1c.nii.gz",
        "/home/pmilab/Xinyang/data/BraTS2023/ASNR-MICCAI-BraTS2023-PED-Challenge-ValidationData/BraTS-PED-00210-000/BraTS-PED-00210-000-t2f.nii.gz",
        "/home/pmilab/Xinyang/data/BraTS2023/ASNR-MICCAI-BraTS2023-PED-Challenge-ValidationData/BraTS-PED-00210-000/BraTS-PED-00210-000-t1n.nii.gz",
        "/home/pmilab/Xinyang/data/BraTS2023/ASNR-MICCAI-BraTS2023-PED-Challenge-ValidationData/BraTS-PED-00210-000/BraTS-PED-00210-000-t2w.nii.gz",
    ]]
    example_2 =[
    [
        "/home/pmilab/Xinyang/data/BraTS2023/ASNR-MICCAI-BraTS2023-PED-Challenge-ValidationData/BraTS-PED-00208-000/BraTS-PED-00208-000-t1c.nii.gz",
        "/home/pmilab/Xinyang/data/BraTS2023/ASNR-MICCAI-BraTS2023-PED-Challenge-ValidationData/BraTS-PED-00208-000/BraTS-PED-00208-000-t2f.nii.gz",
        "/home/pmilab/Xinyang/data/BraTS2023/ASNR-MICCAI-BraTS2023-PED-Challenge-ValidationData/BraTS-PED-00208-000/BraTS-PED-00208-000-t1n.nii.gz",
        "/home/pmilab/Xinyang/data/BraTS2023/ASNR-MICCAI-BraTS2023-PED-Challenge-ValidationData/BraTS-PED-00208-000/BraTS-PED-00208-000-t2w.nii.gz",
    ]]

    #     example_1 = [
    #     "/home/pmilab/Xinyang/data/BraTS2023/ASNR-MICCAI-BraTS2023-PED-Challenge-ValidationData/BraTS-PED-00210-000/",
    # ]
    gr.HTML(value=f"<center><font size='2'> The software provided 'as is', without any warranty or liability.  For research use only and not intended for medical diagnosis. We do not store or access any information uploaded to the platform.</font></center>")
    gr.Examples(
        examples=[example_1, example_2],
        inputs=[image_folder],
        outputs=[mask_file,out_text],
        fn=main_func,
        cache_examples=False,
        label="Preloaded BraTS 2023 examples"
    )
    
    btn.click(fn=main_func, 
        inputs=[image_folder], outputs=[mask_file, out_text],
    )

    slider_axial.change(
        render_axial,
        inputs=[file_to_render, slider_axial, state_axial],
        outputs=[myimage_axial],
        api_name="hohoho"
    )
    slider_coronal.change(
        render_coronal,
        inputs=[file_to_render, slider_coronal, state_coronal],
        outputs=[myimage_coronal],
        api_name="hohoho"
    )
    slider_saggital.change(
        render_saggital,
        inputs=[file_to_render, slider_saggital, state_saggital],
        outputs=[myimage_saggital],
        api_name="hohoho"
    )

if __name__ == "__main__":
    #demo.queue().launch(auth=("admin@pmilab.cnmc.org", "pmilab2023"),server_name="0.0.0.0")
    demo.queue().launch(server_name="0.0.0.0")
