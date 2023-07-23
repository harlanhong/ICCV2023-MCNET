# Experiment part
#full method
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_addr="0.0.0.0" --master_port=12347 run.py --config config/vox-adv-256-3down.yaml --name ExpendMemoryUnitV54_kp15_Unet_Generator_keypoint_aware_run2 --batchsize 8 --kp_num 15 --generator Unet_Generator_keypoint_aware --GFM GeneratorFullModel --memsize 1 --kp_distance 10 --feat_consistent 10 --generator_gan 0 --mbunit ExpendMemoryUnitV54

CUDA_VISIBLE_DEVICES=0 python evaluation_code.py --config config/vox-adv-256-3down.yaml --checkpoint checkpoints/00000099-checkpoint.pth.tar --adapt_scale --kp_num 15 --generator Unet_Generator_keypoint_aware --memsize 1 --mbunit ExpendMemoryUnitV54 --mode relative --csv data/vox_cross_id_evaluation_best_frame.csv --token vox_cross_id_relative

python demo.py  --config config/vox-adv-256-3down.yaml --driving_video /data/fhongac/origDataset/vox1/train/id10686#zDkgVesX7NU#001423#001797.mp4 --checkpoint checkpoints/00000099-checkpoint.pth.tar --relative --adapt_scale --kp_num 15 --generator Unet_Generator_keypoint_aware --result_video synthetic_2.mp4  --source_image /data/fhongac/origDataset/vox1_frames/train/id10686#zDkgVesX7NU#001423#001797.mp4/0000000.png --mbunit ExpendMemoryUnitV54 --memsize 1 
