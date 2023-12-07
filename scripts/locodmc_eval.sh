task_name=('finger_spin') # TwoArmPegInHole Door Lift
frames=1000000
# feature_dim=256
# sgqn_quantile=0.93
# aux_lr=8e-5
action_repeat=2
feature_dim=256
save_snapshot=False
use_wandb=False
nstep=3

for ((i=1;i<3;i+=1));
do
CUDA_VISIBLE_DEVICES=$((i-1))  python locodmc_eval.py \
              task=${task_name} \
              seed=$((i)) \
              action_repeat=${action_repeat} \
              use_wandb=${use_wandb} \
              use_tb=False \
              num_train_frames=${frames} \
              save_snapshot=${save_snapshot} \
              save_video=False \
              feature_dim=${feature_dim} \
              nstep=${nstep} \
              wandb_group=$1 &
done

CUDA_VISIBLE_DEVICES=6  python locodmc_eval.py \
              task=${task_name} \
              seed=3 \
              action_repeat=${action_repeat} \
              use_wandb=${use_wandb} \
              use_tb=False \
              num_train_frames=${frames} \
              save_snapshot=${save_snapshot} \
              save_video=False \
              feature_dim=${feature_dim} \
              nstep=${nstep} \
              wandb_group=$1