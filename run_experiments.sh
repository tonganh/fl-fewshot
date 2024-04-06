# python main.py --task mnist_cnum10_dist1_skew0.8_seed0 --model cnn --algorithm fedtheo --num_rounds 100 --num_epochs 5 --learning_rate 0.001 --proportion 1 --batch_size 10 --eval_interval 1 --gpu 3 --num_threads 1
# python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --model resnet18 --algorithm fedavg --num_rounds 10 --num_epochs 5 --learning_rate 0.001 --proportion 1 --batch_size 10 --eval_interval 1 --gpu 0 --num_threads 1
# python main.py --task mnist_cnum10_dist1_skew0.8_seed0 --model cnn --algorithm fedavg --num_rounds 100 --num_epochs 5 --learning_rate 0.001 --proportion 1 --batch_size 10 --eval_interval 1 --gpu 3
# python main.py --task mnist_cnum10_dist1_skew0.8_seed0 --model cnn --algorithm fedprox --num_rounds 100 --num_epochs 5 --learning_rate 0.001 --proportion 1 --batch_size 10 --eval_interval 1 --gpu 3

# python main.py --task mnist_cnum100_dist2_skew0.3_seed0 --model cnn --algorithm fedrl --num_rounds 100 --num_epochs 5 --learning_rate 0.001 --proportion 1 --batch_size 10 --eval_interval 1 --gpu 3
# python main.py --task mnist_cnum100_dist2_skew0.3_seed0 --model cnn --algorithm fedavg --num_rounds 100 --num_epochs 5 --learning_rate 0.001 --proportion 1 --batch_size 10 --eval_interval 1 --gpu 3
# python main.py --task mnist_cnum100_dist2_skew0.3_seed0 --model cnn --algorithm fedprox --num_rounds 100 --num_epochs 5 --learning_rate 0.001 --proportion 1 --batch_size 10 --eval_interval 1 --gpu 3

# python main.py --task mnist_cnum100_dist3_skew0.4_seed0 --model cnn --algorithm fedrl --num_rounds 100 --num_epochs 5 --learning_rate 0.001 --proportion 1 --batch_size 10 --eval_interval 1 --gpu 3
# python main.py --task mnist_cnum100_dist2_skew0.4_seed0 --model cnn --algorithm fedavg --num_rounds 100 --num_epochs 5 --learning_rate 0.001 --proportion 1 --batch_size 10 --eval_interval 1 --gpu 3
# python main.py --task mnist_cnum100_dist2_skew0.4_seed0 --model cnn --algorithm fedprox --num_rounds 100 --num_epochs 5 --learning_rate 0.001 --proportion 1 --batch_size 10 --eval_interval 1 --gpu 3
python main.py --task cifar100_cnum70_dist9_skew0.8_seed0 --data_split client_data.json --root_data benchmark/cifar100/data --model resnet18_fewshot --algorithm fs_fl --num_rounds 10 --num_train_steps 200 --learning_rate 0.001 --proportion 1 --gpu 0 --num_threads 10