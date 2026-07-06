import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import argparse
import numpy as np
import torch
import json

from core.client import Client
from core.server import Server
from utils.mat_loader import MatDataset
from sklearn.cluster import KMeans
from utils.evaluation import evaluate

def inference(dataset, clients, server, device, seed):
    for c in clients: c.eval()
    server.eval()

    with torch.no_grad():
        idx = torch.arange(len(dataset), device=device)
        z_list = [client(idx) for client in clients]
        z_fused = torch.mean(torch.stack(z_list), dim=0)

    for c in clients: c.train()
    server.train()

    full_labels = dataset.labels
    kmeans = KMeans(n_clusters=len(np.unique(full_labels)), n_init=10, random_state=seed)
    preds = kmeans.fit_predict(z_fused.cpu().numpy())
    return preds, full_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST_USPS")
    parser.add_argument("--instance_temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--dataset_dir", type=str, default="./datasets")
    parser.add_argument("--model_path", type=str, default="save")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--feature_dim", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--topk", type=int, default=15)
    parser.add_argument("--exp_name", type=str, default="")
    
    args = parser.parse_args()    
    
    if args.seed < 0:
        args.seed = int.from_bytes(os.urandom(4), "little")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.model_path): os.makedirs(args.model_path)
    
    mat_path = os.path.join(args.dataset_dir, f"{args.dataset}.mat")
    dataset = MatDataset(mat_path, device=device)
    num_views = len(dataset.views)
    args.class_num = len(np.unique(dataset.labels))

    full_idx = torch.arange(len(dataset), device=device)

    clients = []
    schedulers = []
    for i in range(num_views):
        input_dim = dataset.views[i].shape[1]
        full_data = dataset.views[i]
        client = Client(args, device, input_dim=input_dim, full_data=full_data)
        clients.append(client)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(client.optimizer, T_max=args.epochs, eta_min=1e-5)
        schedulers.append(sched)

    server = Server(args, device)
    server_optimizer = torch.optim.Adam(server.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    server_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(server_optimizer, T_max=args.epochs, eta_min=1e-5)

    print(f">>> Start Training: LR={args.learning_rate}, Temp={args.instance_temperature}, Seed={args.seed}")
    
    results_history = []
    
    for epoch in range(args.start_epoch, args.epochs):
        z_list = []
        for client in clients:
            client.optimizer.zero_grad()
            z = client.forward(full_idx)
            z_list.append(z)

        server_optimizer.zero_grad()
        loss, grads_list = server.step(z_list)
        server_optimizer.step()

        for client, grad in zip(clients, grads_list):
            client.backward(grad)
            torch.nn.utils.clip_grad_norm_(client.parameters(), max_norm=5.0)
            client.optimizer.step()

        for sched in schedulers: sched.step()
        server_scheduler.step()

        if (epoch + 1) % 10 == 0:
            avg_loss = loss
            preds, labels = inference(dataset, clients, server, device, args.seed)
            nmi, ari, f, acc, purity = evaluate(labels, preds)
            
            curr_lr = server_optimizer.param_groups[0]['lr']
            
            print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.4f} | LR: {curr_lr:.6f}")
            print(f"Epoch {epoch+1} | ACC: {acc:.4f} | NMI: {nmi:.4f} | ARI: {ari:.4f} | Purity: {purity:.4f}")

            results_history.append({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "acc": float(acc),
                "nmi": float(nmi),
                "ari": float(ari),
                "purity": float(purity),
                "lr": curr_lr
            })

    if args.exp_name:
        save_dir = os.path.join("result", args.dataset, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/results.json", 'w', encoding='utf-8') as f:
            json.dump({"history": results_history, "dataset": args.dataset}, f, indent=2, ensure_ascii=False)
        print(f"result saved in: {save_dir}/results.json")

if __name__ == "__main__":
    main()
