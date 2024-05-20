python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split data_split/cifar100/dirichlet_1.5_drop.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fed_fewshot \
    --num_rounds 200 --num_train_steps 5 --num_val_steps 200 \
    --learning_rate 0.00001 --proportion 0.2 --gpu 0 --num_threads 1 --num_loader_workers 1 \
    --use_wandb_logging --prototype_loss_weight 100.0 --eval_interval 5 --client_model_aggregation entropy