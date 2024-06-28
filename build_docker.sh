 source ../ISBI2024-BraTS-GOAT/.mlcubes/bin/activate
cd cnmc_brats23/mlcube
mlcube configure -Pdocker.build_strategy=always
docker push aparida12/brats-peds-2023:ped