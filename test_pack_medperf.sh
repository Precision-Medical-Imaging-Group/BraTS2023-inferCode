cd cnmc_brats23/mlcube
mlcube configure -Pdocker.build_strategy=always
docker push docker.synapse.org/syn52117469/cnmc-peds:ped
medperf test run --demo_dataset_url synapse:syn52276402 --demo_dataset_hash "16526543134396b0c8fd0f0428be7c96f2142a66" -p /home/abhijeet/Code/BraTS2023-inferCode/test_mlcubes/prep_segmentation -m /home/abhijeet/Code/BraTS2023-inferCode/cnmc_brats23/mlcube -e /home/abhijeet/Code/BraTS2023-inferCode/test_mlcubes/eval_segmentation --offline --no-cache
cd ../../medperf
python scripts/package-mlcube.py --mlcube /home/abhijeet/Code/BraTS2023-inferCode/cnmc_brats23/mlcube --mlcube-types model --output ./ml-cube-infer-eng.tar.gz