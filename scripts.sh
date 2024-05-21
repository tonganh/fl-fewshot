python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split client_data.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fs_fl --num_rounds 10 --num_train_steps 200 --learning_rate 0.001 --proportion 0.1 --gpu 0 --num_threads 1 --num_loader_workers 1
python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split client_data.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fs_fl --num_rounds 10 --num_train_steps 200 --learning_rate 0.001 --proportion 0.1 --gpu 0 --num_threads 2 --num_loader_workers 1
python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split client_data.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fs_fl --num_rounds 10 --num_train_steps 200 --learning_rate 0.001 --proportion 0.1 --gpu 0 --num_threads 3 --num_loader_workers 1

python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split client_data.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fed_fewshot \
    --num_rounds 100 --num_train_steps 200 --num_val_steps 200 \
    --learning_rate 0.001 --proportion 1.0 --gpu 0 --num_threads 4 --num_loader_workers 1 \
    --use_wandb_logging 

# Anh Tong Anh giup em chay may script nay
python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split client_data.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fed_fewshot \
    --num_rounds 100 --num_train_steps 200 --num_val_steps 200 \
    --learning_rate 0.0001 --proportion 1.0 --gpu 0 --num_threads 4 --num_loader_workers 1 \
    --use_wandb_logging --prototype_loss_weight 0 

python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split client_data.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fed_fewshot \
    --num_rounds 100 --num_train_steps 200 --num_val_steps 200 \
    --learning_rate 0.0001 --proportion 1.0 --gpu 0 --num_threads 4 --num_loader_workers 1 \
    --use_wandb_logging --prototype_loss_weight 0 --use_lrscheduler

python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split client_data.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fed_fewshot \
    --num_rounds 100 --num_train_steps 200 --num_val_steps 200 \
    --learning_rate 0.0001 --proportion 0.05 --gpu 0 --num_threads 4 --num_loader_workers 1 \
    --use_wandb_logging --prototype_loss_weight 0 --debug

python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split client_data.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fed_fewshot \
    --num_rounds 100 --num_train_steps 200 --num_val_steps 200 \
    --learning_rate 0.00001 --proportion 1.0 --gpu 0 --num_threads 4 --num_loader_workers 1 \
    --use_wandb_logging --prototype_loss_weight 0

python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split client_data.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fed_fewshot \
    --num_rounds 100 --num_train_steps 200 --num_val_steps 200 \
    --learning_rate 0.001 --proportion 1.0 --gpu 0 --num_threads 4 --num_loader_workers 1 \
    --use_wandb_logging --prototype_loss_weight 0.01 



python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split client_data.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fed_fewshot \
    --num_rounds 100 --num_train_steps 200 --num_val_steps 200 \
    --learning_rate 0.00001 --proportion 1.0 --gpu 0 --num_threads 4 --num_loader_workers 1 \
    --use_wandb_logging --prototype_loss_weight 0.0 

python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split client_data.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fed_fewshot \
    --num_rounds 100 --num_train_steps 200 --num_val_steps 200 \
    --learning_rate 0.00001 --proportion 1.0 --gpu 0 --num_threads 4 --num_loader_workers 1 \
    --use_wandb_logging --prototype_loss_weight 0.1 

python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split client_data.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fed_fewshot \
    --num_rounds 100 --num_train_steps 200 --num_val_steps 200 \
    --learning_rate 0.0001 --proportion 1.0 --gpu 0 --num_threads 4 --num_loader_workers 1 \
    --use_wandb_logging --prototype_loss_weight 0.0 

python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split client_data.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fed_fewshot \
    --num_rounds 100 --num_train_steps 200 --num_val_steps 200 \
    --learning_rate 0.0001 --proportion 1.0 --gpu 0 --num_threads 4 --num_loader_workers 1 \
    --use_wandb_logging --prototype_loss_weight 0.1 

python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split client_data.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fed_fewshot \
    --num_rounds 100 --num_train_steps 200 --num_val_steps 200 \
    --learning_rate 0.000001 --proportion 1.0 --gpu 0 --num_threads 4 --num_loader_workers 1 \
    --use_wandb_logging --prototype_loss_weight 0.0 

python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split client_data.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fed_fewshot \
    --num_rounds 100 --num_train_steps 200 --num_val_steps 200 \
    --learning_rate 0.000001 --proportion 1.0 --gpu 0 --num_threads 4 --num_loader_workers 1 \
    --use_wandb_logging --prototype_loss_weight 0.1 

    srun --nodelist=slurmnode1 --pty bash -i

python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split client_data.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fed_fewshot \
    --num_rounds 200 --num_train_steps 10 --num_val_steps 200 \
    --learning_rate 0.000001 --proportion 1.0 --gpu 0 --num_threads 4 --num_loader_workers 1 \
    --use_wandb_logging --prototype_loss_weight 0.0 


# new script
python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split client_data.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fed_fewshot \
    --num_rounds 300 --num_train_steps 5 --num_val_steps 200 \
    --learning_rate 0.00001 --proportion 1.0 --gpu 0 --num_threads 4 --num_loader_workers 1 \
    --use_wandb_logging --prototype_loss_weight 0 --eval_interval 5

python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split client_data.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fed_fewshot \
    --num_rounds 300 --num_train_steps 5 --num_val_steps 200 \
    --learning_rate 0.00001 --proportion 1.0 --gpu 0 --num_threads 4 --num_loader_workers 1 \
    --use_wandb_logging --prototype_loss_weight 1.0 --eval_interval 5

python generate_split.py --dist 1 --save_path data_split/cifar100/dirichlet_0.5.json
python generate_split.py --dist 1 --save_path data_split/cifar100/dirichlet_0.1.json --skewness 0.1


# dirichlet 
python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split datasplit_dirichlet.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fed_fewshot \
    --num_rounds 300 --num_train_steps 5 --num_val_steps 200 \
    --learning_rate 0.00001 --proportion 1.0 --gpu 0 --num_threads 4 --num_loader_workers 1 \
    --use_wandb_logging --prototype_loss_weight 0 --eval_interval 5 --client_model_aggregation entropy 

python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split datasplit_dirichlet.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fed_fewshot \
    --num_rounds 300 --num_train_steps 5 --num_val_steps 200 \
    --learning_rate 0.00001 --proportion 1.0 --gpu 0 --num_threads 4 --num_loader_workers 1 \
    --use_wandb_logging --prototype_loss_weight 1.0 --eval_interval 5 --client_model_aggregation entropy

python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split data_split/cifar100/dirichlet_0.1.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fed_fewshot \
    --num_rounds 300 --num_train_steps 5 --num_val_steps 200 \
    --learning_rate 0.00001 --proportion 1.0 --gpu 0 --num_threads 2 --num_loader_workers 1 \
    --use_wandb_logging --prototype_loss_weight 0 --eval_interval 5 --client_model_aggregation entropy 

python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split data_split/cifar100/dirichlet_0.1.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fed_fewshot \
    --num_rounds 300 --num_train_steps 5 --num_val_steps 200 \
    --learning_rate 0.00001 --proportion 1.0 --gpu 0 --num_threads 4 --num_loader_workers 1 \
    --use_wandb_logging --prototype_loss_weight 1.0 --eval_interval 5 --client_model_aggregation entropy

python generate_split.py --dist 1 --save_path data_split/cifar100/dirichlet_0.5_drop.json --skewness 0.5 --drop_if_lack_data
python generate_split.py --dist 1 --save_path data_split/cifar100/dirichlet_1.5_drop.json --skewness 1.5 --drop_if_lack_data

# adjust prototypical loss weight
python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split data_split/cifar100/dirichlet_1.5_drop.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fed_fewshot \
    --num_rounds 200 --num_train_steps 5 --num_val_steps 200 \
    --learning_rate 0.00001 --proportion 1.0 --gpu 0 --num_threads 4 --num_loader_workers 1 \
    --use_wandb_logging --prototype_loss_weight 0.0 --eval_interval 5 --client_model_aggregation entropy

python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split data_split/cifar100/dirichlet_1.5_drop.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fed_fewshot \
    --num_rounds 200 --num_train_steps 5 --num_val_steps 200 \
    --learning_rate 0.00001 --proportion 1.0 --gpu 0 --num_threads 4 --num_loader_workers 1 \
    --use_wandb_logging --prototype_loss_weight 10.0 --eval_interval 5 --client_model_aggregation entropy

python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split data_split/cifar100/dirichlet_1.5_drop.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fed_fewshot \
    --num_rounds 200 --num_train_steps 5 --num_val_steps 200 \
    --learning_rate 0.00001 --proportion 1.0 --gpu 0 --num_threads 4 --num_loader_workers 1 \
    --use_wandb_logging --prototype_loss_weight 100.0 --eval_interval 5 --client_model_aggregation entropy

python generate_split.py --dist 0 --num_clients 70 --save_path data_split/cifar100/iid_c70.json 

python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split data_split/cifar100/iid_c70.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fed_fewshot \
    --num_rounds 200 --num_train_steps 5 --num_val_steps 200 \
    --learning_rate 0.00001 --proportion 0.5 --gpu 0 --num_threads 4 --num_loader_workers 1 \
    --use_wandb_logging --prototype_loss_weight 0.0 --eval_interval 5 --client_model_aggregation entropy

python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split data_split/cifar100/iid_c70.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fed_fewshot \
    --num_rounds 200 --num_train_steps 5 --num_val_steps 200 \
    --learning_rate 0.00001 --proportion 0.5 --gpu 0 --num_threads 4 --num_loader_workers 1 \
    --use_wandb_logging --prototype_loss_weight 10.0 --eval_interval 5 --client_model_aggregation entropy