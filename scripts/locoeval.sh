CUDA_VISIBLE_DEVICES=5 python3 locoeval.py \
  --algorithm $1 \
	--eval_episodes 100 \
	--seed 5 \
	--domain_name $2 \
	--task_name $3 \
	--model_dir /to/your/path