import argparse
import json
import torch
from torchvision import datasets, transforms

from benchmark.cifar100.model.resnet18_fewshot import Model
from benchmark.toolkits import XYDatasetFewShot
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F
import wandb
import os
from dotenv import load_dotenv
from tqdm import tqdm

# Load variables from .env file
load_dotenv()
def get_total_data(data_path):
    train_data = datasets.CIFAR100(
        data_path,
        train=True,
        download=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    )
    test_data = datasets.CIFAR100(
        data_path,
        train=False,
        download=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        ),
    )
    dataset = ConcatDataset([train_data, test_data])
    return dataset

def compute_loss(output, input):
    return torch.nn.CrossEntropyLoss()(output, input)

def accuracy(logits, target_labels, mode="mean"):
    preds = torch.argmax(logits, -1)
    acc = torch.sum((preds == target_labels).float()) / target_labels.shape[0]
    return acc
def prepare_input(input, device):
    for key in input.keys():
        if isinstance(input[key], torch.Tensor):
            input[key] = input[key].to(device)
            input[key] = input[key].squeeze(0)
    input["query_labels"] = input["query_labels"].to(dtype=torch.long)
    return input

@torch.no_grad()
def eval(args, model, test_loader, device):
    model.eval()
    num_iters = args.num_val_iters
    loader_iter = iter(test_loader)

    running_loss = []
    running_acc = []
    
    for i in tqdm(range(num_iters)):
        try:
            input = next(loader_iter)
        except StopIteration:
            loader_iter = iter(test_loader)
            input = next(loader_iter)
        
        input = prepare_input(input, device)
        output = model(input)
        
        loss = F.cross_entropy(output['logits'], input['query_labels'])
        acc = (output['logits'].argmax(1) == input['query_labels']).float().mean()

        running_loss.append(loss.item())
        running_acc.append(acc.item())
    
    return {
        'loss': sum(running_loss) / len(running_loss),
        'acc': sum(running_acc) / len(running_acc)
    }


def train(args, model, train_loader, test_loader, optimizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    optimizer.zero_grad()
    num_iters = args.num_train_iters
    grad_accum = args.grad_accum
    loader_iter = iter(train_loader)
    
    running_loss = []
    running_acc = []
    for i in tqdm(range(num_iters)):
        wandb_log = {}
        try:
            input = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            input = next(loader_iter)
        
        input = prepare_input(input, device)
        output = model(input)
        
        loss = F.cross_entropy(output['logits'], input['query_labels'])
        loss.backward()
        acc = (output['logits'].argmax(1) == input['query_labels']).float().mean()

        running_loss.append(loss.item())
        running_acc.append(acc.item())
        
        if (i + 1) % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        if (i + 1) % args.val_interval == 0:
            res = eval(args, model, test_loader, device)
            wandb_log['val_acc'] = res['acc']
            wandb_log['val_loss'] = res['loss']
            model.train()
        
        if (i + 1) % args.log_interval == 0:
            wandb_log[f'train_loss_{args.log_interval}'] = sum(running_loss) / len(running_loss)
            wandb_log[f'train_acc_{args.log_interval}'] = sum(running_acc) / len(running_acc)
            running_loss = []
            running_acc = []
        
        if not args.debug:
            wandb.log(wandb_log)

def init_session(args):
    if not args.debug:
        load_dotenv()
        wandb_key = os.getenv('WANDB_KEY')
        wandb.login(key=wandb_key)
        wandb.init(project='fed_fewshot', entity='aiotlab')
        wandb.config.update(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='benchmark/cifar100/data')
    parser.add_argument('--data_split_path', type=str, default='data_split/cifar100/iid_c2.json')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--num_train_iters', type=int, default=5000)
    parser.add_argument('--num_val_iters', type=int, default=200)
    parser.add_argument('--val_interval', type=int, default=50)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--grad_accum', type=int, default=4)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    init_session(args)

    total_data = get_total_data(args.data_path)
    with open(args.data_split_path, 'r') as f:
        split = json.load(f)
    # dataset
    train_dataset = XYDatasetFewShot(total_data, split['client_data'][0]['train'], split['client_data'][0]['train_labels'])
    test_dataset = XYDatasetFewShot(total_data, split['test_data_ids'], split['test_data_labels'])
    # dataloader
    train_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=args.num_workers
    )
    # model
    model = Model()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # train loop
    train(args, model, train_loader, test_loader, optimizer)
