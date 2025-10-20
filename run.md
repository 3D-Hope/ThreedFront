
<!-- Pickling Data -->
python scripts/pickle_threed_front_dataset.py /mnt/sv-share/3DFRONT/data/3D-FRONT /mnt/sv-share/3DFRONT/data/3D-FUTURE-model /mnt/sv-share/3DFRONT/data/3D-FUTURE-model/model_info.json

python scripts/pickle_threed_future_dataset.py threed_front_bedroom

python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-09-27/10-12-32/sampled_scenes_results.pkl
python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-09-27/10-15-17/sampled_scenes_results.pkl
<!-- Preprocessing -->

python scripts/preprocess_data.py threed_front_bedroom

python scripts/preprocess_data.py threed_front_livingroom

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
source ../steerable-scene-generation/.venv/bin/activate

python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-09-27/10-12-32/sampled_scenes_results.pkl --no_texture --without_floor
python ../ThreedFront/scripts/render_results.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-09-27/10-15-17/sampled_scenes_results.pkl --no_texture --without_floor
<!-- FID -->
python scripts/compute_fid_scores.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-16/10-45-35/sampled_scenes_results.pkl --output_directory ./fid_tmps --no_texture --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/ --no_floor

python scripts/compute_fid_scores.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-11/14-35-27/sampled_scenes_results.pkl --output_directory ./fid_tmps --no_texture --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/ --no_floor
<!-- KID -->
python scripts/compute_fid_scores.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-16/10-45-35/sampled_scenes_results.pkl --compute_kid --output_directory ./fid_tmps --no_texture  --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/ --no_floor

python scripts/compute_fid_scores.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-11/14-35-27/sampled_scenes_results.pkl --compute_kid --output_directory ./fid_tmps --no_texture  --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/ --no_floor
<!-- Classifier -->
python scripts/synthetic_vs_real_classifier.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-11/14-35-27/sampled_scenes_results.pkl --output_directory ./classifier_tmps --no_texture --no_floor  --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/

python scripts/synthetic_vs_real_classifier.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-16/10-45-35/sampled_scenes_results.pkl --output_directory ./classifier_tmps --no_texture --no_floor  --dataset_directory /mnt/sv-share/MiDiffusion/gravee/bedroom/
<!-- OOB and MBL -->
source ../steerable-scene-generation/.venv/bin/activate
python scripts/bbox_analysis.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-12/21-03-00/sampled_scenes_results.pkl


python scripts/bbox_analysis.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-11/14-35-27/sampled_scenes_results.pkl

python scripts/bbox_analysis.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-16/10-45-35/sampled_scenes_results.pkl

<!-- KL-Divergence -->
python scripts/evaluate_kl_divergence_object_category.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-16/10-45-35/sampled_scenes_results.pkl --output_directory ./kl_tmps

python scripts/evaluate_kl_divergence_object_category.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-11/14-35-27/sampled_scenes_results.pkl --output_directory ./kl_tmps

python scripts/evaluate_kl_divergence_object_category.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-10-16/10-45-35/sampled_scenes_results.pkl --output_directory ./kl_tmps
<!-- Obj Metric -->
python scripts/calculate_num_obj.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-09-27/10-12-32/sampled_scenes_results.pkl
python scripts/calculate_num_obj.py /media/ajad/YourBook/AshokSaugatResearchBackup/AshokSaugatResearch/steerable-scene-generation/outputs/2025-09-27/10-15-17/sampled_scenes_results.pkl
