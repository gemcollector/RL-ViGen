task_name='door'
save_snapshot=False
use_wandb=False
stage1_use_pretrain=False
num_train_frames=1001000
stage2_n_update=0
save_models=True
local_data_dir=/to/your/path


python train_adroit_dhm.py \
		task=${task_name} \
		seed=5 \
		stage1_use_pretrain=${stage1_use_pretrain} \
		save_snapshot=${save_snapshot} \
		device=cuda:0 \
		local_data_dir=${local_data_dir} \
		use_wandb=${use_wandb} \
		stage2_n_update=${stage2_n_update} \
		num_train_frames=${num_train_frames} \
		save_models=${save_models} \
		wandb_group=$2