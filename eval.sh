save_snapshot=False
use_wandb=False
env='carla'

if [ "$env" = "robosuite" ]; then
task_name='TwoArmPegInHole'
test_agent='test_pieg'
action_repeat=1
CUDA_VISIBLE_DEVICES=1  python eval.py \
              env=${env} \
              task=${task_name} \
              seed=5 \
              action_repeat=${action_repeat} \
              use_wandb=${use_wandb} \
              use_tb=False \
              save_snapshot=${save_snapshot} \
              save_video=False \
              wandb_group=${test_agent}
elif [ "$env" = "habitat" ]; then
  test_agent='test_pieg_'
  action_repeat=2
  for ((i=1;i<11;i+=1))
  do
  CUDA_VISIBLE_DEVICES=7  python eval.py \
                env=${env} \
                task@_global_='habitat' \
                seed=3 \
                action_repeat=${action_repeat} \
                use_wandb=${use_wandb} \
                use_tb=False \
                save_video=False \
                save_snapshot=${save_snapshot} \
                wandb_group=${test_agent}$((i))
  done
elif [ "$env" = "carla" ]; then
task_name='carla'
test_agent='test_pieg'
action_repeat=2

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 CUDA_VISIBLE_DEVICES=1  python eval.py \
                                                                    env=${env} \
                                                                    task=${task_name} \
                                                                    seed=5 \
                                                                    action_repeat=${action_repeat} \
                                                                    use_wandb=${use_wandb} \
                                                                    use_tb=False \
                                                                    save_snapshot=${save_snapshot} \
                                                                    save_video=False \
                                                                    wandb_group=${test_agent}

fi