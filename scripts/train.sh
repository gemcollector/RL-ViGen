#task_name=unitree_stand
#frames=1001000
##CUDA_VISIBLE_DEVICES=7 python train.py \
##              task=${task_name} \
##              seed=3 \
##              use_tb=False \
##              num_train_frames=${frames} \
##              save_snapshot=True \
##              use_wandb=False &
##
#CUDA_VISIBLE_DEVICES=4 python train.py \
#              task=${task_name} \
#              seed=2 \
#              use_tb=False \
#              num_train_frames=${frames} \
#              save_snapshot=False \
#              use_wandb=False
#
#CUDA_VISIBLE_DEVICES=5 python train.py \
#              task=${task_name} \
#              seed=1 \
#              use_tb=False \
#              num_train_frames=${frames} \
#              save_snapshot=True \
#              use_wandb=False


easy_task_list=('unitree_walk')
frames=1001000
feature_dim=50
sgqn_quantile=0.93
action_repeat=2
aux_lr=8e-5
env=dmc

for task_name in ${easy_task_list[@]};
do
#	for ((i=1;i<5;i+=1))
#	do
#		CUDA_VISIBLE_DEVICES=$((i+3)) python train.py \
#					task=${task_name} \
#					seed=$((i)) \
#          action_repeat=${action_repeat} \
#					use_wandb=False \
#					num_train_frames=${frames} \
#					use_tb=False \
#					save_video=False \
#					feature_dim=${feature_dim} \
#					save_snapshot=True  &
##					agent.sgqn_quantile=${sgqn_quantile} \
##          agent.aux_lr=${aux_lr} &
#	done

	CUDA_VISIBLE_DEVICES=0  python train.py \
								env=${env} \
								task=${task_name} \
								seed=5 \
								action_repeat=${action_repeat} \
								use_wandb=False \
								use_tb=False \
								num_train_frames=${frames} \
								save_snapshot=True \
								save_video=False \
								feature_dim=${feature_dim}
# 							  agent.sgqn_quantile=${sgqn_quantile} \
# 							  agent.aux_lr=${aux_lr}
done