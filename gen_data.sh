# python generate_fedtask.py --dataset mnist --dist 0 --skew 0 --num_clients 100
rm -rf fedtask && python generate_fedtask.py --dataset cifar100 --dist 9 --skew 0.8 --num_clients 70 --number_class_per_client 20
# python generate_fedtask.py --dataset mnist --dist 2 --skew 0.8 --num_clients 10
# python generate_fedtask.py --dataset mnist --dist 3 --skew 0.8 --num_clients 10