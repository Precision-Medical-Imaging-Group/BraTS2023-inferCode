# BraTS2023 Inference Code

This repository contains the final code submitted for the [MICCAI2023](https://conferences.miccai.org/2023) Brain Tumor Segmentation Challenge ([BraTS2023](https://www.synapse.org/#!Synapse:syn51156910/wiki/621282)) by the team CNMC_PMI.
Our team achieved top rankings in the BraTS2023 challenge:
- **1st place** in the PED task.
- **3rd place** in the MEN task.
- **4th place** in the MET task.

## Getting Started
![Python 3.9](https://img.shields.io/badge/python-3.9-green.svg)
[![monai 1.3](https://img.shields.io/badge/monai-1.3-cyan.svg)](https://monai.io/)

### Prerequisites
- **Important**: Use Python 3.9 only.
### Environment Setup
Create and activate a virtual environment:
```bash
python3.9 -m venv .mlcubes
source .mlcubes/bin/activate
```
### Installation
Install and log into MedPerf:
```bash
sh install_medperf.sh
```
### Access to Trained Models
All trained model weights are accessible to authorized individuals here: [OneDrive-Weights](https://cnmc-my.sharepoint.com/:f:/g/personal/pabhijeet_childrensnational_org/EmMNukOtj3REgw5EM4AWjnABX_IPNtO2myg6LSdtAmUySw?e=SuWRtc)

### MLCubes, MedPerfs and Docker Integration
- For the PED task, visit the [mlperf branch](https://github.com/Precision-Medical-Imaging-Group/BraTS2023-inferCode/tree/mlperf)
- For the MEN task, visit the [men-mlperf branch](https://github.com/Precision-Medical-Imaging-Group/BraTS2023-inferCode/tree/men-mlperf)
- For the MET task, visit the [met-mlperf branch](https://github.com/Precision-Medical-Imaging-Group/BraTS2023-inferCode/tree/met-mlperf)

## Contact Us

For momre information about implementations and reusability, please contact 
- Abhijeet Parida [pabhijeet@childrensnational.org](mailto:pabhijeet@childrensnational.org)
- Daniel Capell&aacute;n [daniel.capellan@upm.es](mailto:daniel.capellan@upm.es)
- Zhifan Jiang, Ph.D. [zjiang@childrensnational.org](mailto:zjiang@childrensnational.org)
