easy_task_list=('carla')
frames=1001000
# feature_dim=256
sgqn_quantile=0.9
aux_lr=8e-5
action_repeat=2
feature_dim=50
nstep=1
save_snapshot=True
use_wandb=False
lr=1e-4

for task_name in ${easy_task_list[@]};
do
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7	CUDA_VISIBLE_DEVICES=5  python carlatrain.py \
                                                        task=${task_name} \
                                                        seed=3 \
                                                        action_repeat=${action_repeat} \
                                                        use_wandb=${use_wandb} \
                                                        use_tb=False \
                                                        num_train_frames=${frames} \
                                                        save_snapshot=${save_snapshot} \
                                                        save_video=False \
                                                        lr=${lr} \
                                                        feature_dim=${feature_dim} \
                                                        nstep=${nstep} \
                                                        wandb_group=$1
done