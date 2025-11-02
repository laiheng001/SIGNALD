import torch
import torch.nn as nn
from torch_geometric.data import Dataset, DataLoader
from torch_geometric.nn import global_mean_pool
from egnn_clean import EGNN

import os, sys, argparse, pickle
import pandas as pd
from glob import glob
import threading
from queue import Queue

class EGNNModel(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, n_layers, out_node_nf, dropout = 0, act_fn=nn.SiLU()):
        super(EGNNModel, self).__init__()
        # EGNN layer from egnn_clean
        self.egnn = EGNN(residual=True,attention=True,normalize=True,tanh=False,act_fn=act_fn,device=device,
                         in_node_nf=in_node_nf, hidden_nf=hidden_nf, out_node_nf=out_node_nf, in_edge_nf=in_edge_nf, n_layers=n_layers )
        # Output layer for regression
        self.out_layer = nn.Sequential(
            nn.Linear(out_node_nf, hidden_nf),
            act_fn,
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_nf, 1)
        )
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, data):
        x, pos, edge_index, edge_attr, batch = data.x, data.pos, data.edge_index, data.edge_attr, data.batch
        node_feats, pos_out = self.egnn(x, pos, edge_index, edge_attr=edge_attr)
        node_feats = self.dropout(node_feats)
        out = global_mean_pool(node_feats, batch)
        out = self.out_layer(out)
        return out # out

class CustomDataset(Dataset):
    def __init__(self, directories):
        self.directories = directories
        super(CustomDataset, self).__init__()
    def len(self):
        return len(self.directories)
    def get(self, idx):
        file_path = self.directories[idx]
        graph_data = pickle.load(open(file_path, 'rb'))
        graph_data.dir = file_path
        return graph_data
    def get_indices(self):
        return range(self.len())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="python generate_distance_feature.py -inp rdock_allresult_all.csv -cutoff 10 -output_folder 3_distance_rdock_docking -ncpus 8")
    parser.add_argument("-graph", type=str, help = "input folder containing .graph files", default="graph_egnn")
    parser.add_argument("-model", type=str, help = "saved model path", default = "model_full_model_epoch300.pth")
    parser.add_argument("-output", type=str, help = "output score filename", default = "SIGNALD_pred.csv")
    parser.add_argument("-ncpus", type=int, help = "no. of processors", default=8)
    
    args = parser.parse_args()
    
    # Model parameter  ###
    batch_size= 128
    hidden_dim, hidden_layer, out_node = 128, 4, 8
    huber, lr, optimizer, dropout, momentum = 2, 0.0005, "Adam", 0.4, 0

    eval_directory = glob(f"{args.graph}/*/*.graph")
    eval_dataset = CustomDataset(eval_directory)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, num_workers = args.ncpus)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initiate model #
    model = EGNNModel(next(iter(eval_loader)).x.size(1), next(iter(eval_loader)).edge_attr.size(1), hidden_dim, hidden_layer, out_node, dropout)
    model.load_state_dict(torch.load(args.model))
    # model = torch.load(args.model)
    model = model.to(device)
    
    log_queue = Queue()
    stop_token = object()

    def writer_thread_fn(queue, filename):
        with open(filename, "w") as f:
            f.write("Name,prediction,natom\n")
            while True:
                item = queue.get()
                if item is stop_token:
                    break
                f.writelines(item)

    writer_thread = threading.Thread(target=writer_thread_fn, args=(log_queue, args.output))
    writer_thread.start()
    
    model.eval()
    with torch.no_grad():
        for data in eval_loader:
            data = data.to(device)
            predictions = model(data)
            predictions_flat = predictions.detach().cpu().numpy().flatten()
            natoms = data.n.float().view(-1, 1).cpu().numpy().flatten()
            log_entries = [f"{os.path.basename(dir).replace('.graph', '')},{pred},{int(natom)}\n" for dir, pred, natom in zip(data.dir, predictions_flat, natoms)]
            log_queue.put(log_entries)
    
    log_queue.put(stop_token)
    writer_thread.join()
        
    df = pd.read_csv(args.output)
    df["ligand"] = df["Name"].apply(lambda x: x.split("_out")[0])
    df["pred_norm"] = df["prediction"] / df["natom"]
    df.to_csv(args.output, index = False)
