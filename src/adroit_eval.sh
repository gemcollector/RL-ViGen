task_name='door'
save_snapshot=False
use_wandb=False
stage1_use_pretrain=False
num_train_frames=1001000
stage2_n_update=0
save_models=False
model_dir=/to/your/path


python eval_adroit.py \
		task=${task_name} \
		seed=2 \
		stage1_use_pretrain=${stage1_use_pretrain} \
		save_snapshot=${save_snapshot} \
		device=cuda:1 \
		use_wandb=${use_wandb} \
		stage2_n_update=${stage2_n_update} \
		num_train_frames=${num_train_frames} \
		save_models=${save_models} \
		model_dir=${model_dir} \
		wandb_group=$1