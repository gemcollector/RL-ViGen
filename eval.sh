task_name=('TwoArmPegInHole') # TwoArmPegInHole Door Lift
frames=801000
# feature_dim=256
# sgqn_quantile=0.93
# aux_lr=8e-5
action_repeat=1
feature_dim=256
save_snapshot=False
use_wandb=False
nstep=3
env='robosuite'

#for ((i=1;i<5;i+=1));
#do
#CUDA_VISIBLE_DEVICES=$((i+3))  python roboeval.py \
#              task=${task_name} \
#              seed=$((i)) \
#              action_repeat=${action_repeat} \
#              use_wandb=${use_wandb} \
#              use_tb=False \
#              num_train_frames=${frames} \
#              save_snapshot=${save_snapshot} \
#              save_video=False \
#              feature_dim=${feature_dim} \
#              wandb_group=$1 &
#done

CUDA_VISIBLE_DEVICES=1  python eval.py \
              env=${env} \
              task=${task_name} \
              seed=5 \
              action_repeat=${action_repeat} \
              use_wandb=${use_wandb} \
              use_tb=False \
              num_train_frames=${frames} \
              save_snapshot=${save_snapshot} \
              save_video=False \
              feature_dim=${feature_dim} \
              wandb_group=$1