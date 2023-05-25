task_name=habitat # TwoArmPegInHole Door Lift
frames=701000
# feature_dim=256
sgqn_quantile=0.93
aux_lr=8e-5
action_repeat=2
feature_dim=50
save_snapshot=True
use_wandb=False
lr=1e-4
env=habitat



CUDA_VISIBLE_DEVICES=1  python train.py \
                            env=${env} \
                            task@_global_=${task_name} \
                            seed=3 \
                            action_repeat=${action_repeat} \
                            use_wandb=${use_wandb} \
                            use_tb=False \
                            save_video=False \
                            num_train_frames=${frames} \
                            save_snapshot=${save_snapshot} \
                            feature_dim=${feature_dim} \
                            lr=${lr} \
                            wandb_group=$1