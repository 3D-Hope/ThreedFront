
<!-- Pickling Data -->
python scripts/pickle_threed_front_dataset.py /mnt/sv-share/3DFRONT/data/3D-FRONT /mnt/sv-share/3DFRONT/data/3D-FUTURE-model /mnt/sv-share/3DFRONT/data/3D-FUTURE-model/model_info.json

python scripts/pickle_threed_future_dataset.py threed_front_bedroom


<!-- Preprocessing -->

python scripts/preprocess_data.py threed_front_bedroom

<!-- For no texture and no floor (Select 1 if need to replace) --> 
python scripts/preprocess_data.py threed_front_bedroom --no_texture --output_directory /mnt/sv-share/MiData
python scripts/test_preprocess.py threed_front_bedroom --no_texture --output_directory /mnt/sv-share/MiData

`
<!-- For Floorplan Data -->
python scripts/preprocess_floorplan.py /mnt/sv-share/MiData/bedroom --room_side 3.1
python scripts/preprocess_floorplan_cuboid_scene.py /mnt/sv-share/MiData/test_data --room_side 3.1
python scripts/preprocess_floorplan_cuboid_scene.py /mnt/sv-share/MiData/preprocessed_bedrooms_no_walls_objfeat32_unscaled --room_side 3.1


<!-- Test rendering -->
python scripts/render_threedfront_scene.py MasterBedroom-2888 --without_screen --with_walls --with_door_and_windows

<!-- Metrics -->

<!-- FID -->
python scripts/compute_fid_scores.py /home/ajad/AshokSaugatResearch/MiDiffusion/output/predicted_results/baseline_ckpt_20000/results.pkl --output_directory ./fid_tmps --no_texture
<!-- KID -->
python scripts/compute_fid_scores.py /home/ajad/AshokSaugatResearch/MiDiffusion/output/predicted_results/baseline_ckpt_7000/results.pkl --compute_kid --output_directory ./fid_tmps --no_texture
<!-- Classifier -->
python scripts/synthetic_vs_real_classifier.py /home/ajad/AshokSaugatResearch/MiDiffusion/output/predicted_results/baseline_ckpt_20000/results.pkl --output_directory ./classifier_tmps --no_texture
<!-- OOB and MBL -->
python scripts/bbox_analysis.py /home/ajad/AshokSaugatResearch/MiDiffusion/output/predicted_results/baseline_ckpt_20000/results.pkl
<!-- KL-Divergence -->
python scripts/evaluate_kl_divergence_object_category.py /home/ajad/AshokSaugatResearch/MiDiffusion/output/predicted_results/baseline_ckpt_20000/results.pkl --output_directory ./kl_tmps
<!-- Obj Metric -->
python scripts/calculate_num_obj.py /home/ajad/AshokSaugatResearch/MiDiffusion/output/predicted_results/baseline_ckpt_20000/results.pkl
