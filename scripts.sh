python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split client_data.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fs_fl --num_rounds 10 --num_train_steps 200 --learning_rate 0.001 --proportion 0.1 --gpu 0 --num_threads 1 --num_loader_workers 1
python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split client_data.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fs_fl --num_rounds 10 --num_train_steps 200 --learning_rate 0.001 --proportion 0.1 --gpu 0 --num_threads 2 --num_loader_workers 1
python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split client_data.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fs_fl --num_rounds 10 --num_train_steps 200 --learning_rate 0.001 --proportion 0.1 --gpu 0 --num_threads 3 --num_loader_workers 1
python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split client_data.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fed_fewshot --num_rounds 10 --num_train_steps 200 \
    --learning_rate 0.001 --proportion 1.0 --gpu 0 --num_threads 4 --num_loader_workers 1 \
    --use_wandb_logging




