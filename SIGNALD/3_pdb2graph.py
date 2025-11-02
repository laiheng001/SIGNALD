import numpy as np
import pandas as pd
from pymol import cmd
import pickle, os, argparse, math
from glob import glob
from multiprocessing import Pool
import torch
from torch_geometric.data import Data


bond_types = ['1', '2', '3', 'ar', 'inter']
inter_bond = np.zeros(len(bond_types)).astype(int)
inter_bond[bond_types.index("inter")] = 1

def distance(atom1, atom2):
    return np.linalg.norm(atom1 - atom2)

def rgyrate(selection='(all)'):
    model = cmd.get_model(selection).atom
    x = [i.coord for i in model]
    mass = [i.get_mass() for i in model]
    xm = [(m*i,m*j,m*k) for (i,j,k),m in zip(x,mass)]
    tmass = sum(mass)
    rr = sum(mi*i+mj*j+mk*k for (i,j,k),(mi,mj,mk) in zip(x,xm))
    mm = sum((sum(i)/tmass)**2 for i in zip(*xm))
    rg = math.sqrt(rr/tmass - mm)
    return rg

cutoff = 10
def mol2graph(lig):
    try:
        output_folder = os.path.join(args.output_folder, lig)
        os.makedirs(output_folder, exist_ok=True)
        receptor_pdb = receptor_mol2.replace(".mol2", ".pdb")
        df_receptor_bond = pd.read_parquet(receptor_pdb.replace(".pdb", "_bond.parquet"))
        df_receptor_coord = pd.read_parquet(receptor_pdb.replace(".pdb", "_coord.parquet"))
        df_receptor_coord["x"] = df_receptor_coord.apply(lambda row: np.array([row['x_coord'], row['y_coord'], row['z_coord']]), axis = 1)
        df_receptor_atominfo = pd.read_parquet(receptor_pdb.replace(".pdb", "_atominfo.parquet"))
        df_receptor_atominfo["f"] = df_receptor_atominfo.apply(lambda x: list(x["onehot_residue"]) + list(x["onehot_element"]) + list(x["values_rdkit"]),axis = 1)
        
        ligand_directory = os.path.join(args.ligand, lig)
        df_ligand_bond = pd.read_parquet(os.path.join(ligand_directory, lig + "_bond.parquet"))
        df_ligand_atominfo = pd.read_parquet(os.path.join(ligand_directory, lig + "_atominfo.parquet"))
        df_ligand_atominfo["f"] = df_ligand_atominfo.apply(lambda x: list(x["onehot_residue"]) + list(x["onehot_element"]) + list(x["values_rdkit"]), axis = 1)
        cmd.reinitialize()
        cmd.load(os.path.join(ligand_directory, lig + "_out1.pdb"), "ligand")
        cmd.remove("elem H+D")
        neighborbond_list = []
        for ligID in df_ligand_atominfo['atom_number'].values:
            cmd.select("ref", f'ID {ligID}')
            neighbor = cmd.select("refnb","neighbor ref")
            neighbor2 = cmd.select("refnb2","ref extend 2 and not ref and not refnb")
            if neighbor2 >= 1:
                neighbor_space2 = {"IDlist":[]}
                cmd.iterate("refnb2", "IDlist.append(ID)", space = neighbor_space2)
                pair_list = [(ligID,nb2) for nb2 in neighbor_space2["IDlist"]] + [(nb2, ligID) for nb2 in neighbor_space2["IDlist"]]
                neighborbond_list.extend(pair_list)
        neighborbond_list = set(neighborbond_list)
                
        for ligand_pdb in glob(f"{args.ligand}/{lig}/*out*.pdb"):
            df_ligand_coord = pd.read_parquet(ligand_pdb.replace(".pdb", "_coord.parquet"))
            df_ligand_coord["x"] = df_ligand_coord.apply(lambda row: np.array([row['x_coord'], row['y_coord'], row['z_coord']]), axis = 1)
            ### Get nearby receptor atom number ###
            cmd.reinitialize()
            cmd.load(ligand_pdb, "ligand")
            cmd.load(receptor_pdb, "receptor")
            cmd.remove("elem H+D")
            cmd.remove("resn ZN+NCO+EPE")
            cmd.select("nearby", "(ligand around {}) and receptor".format(10))
            nearby_receptor_num = set(cmd.identify("nearby"))
            if len(nearby_receptor_num) == 0:
                with open(error_log, "a") as f:
                    f.write("{} no nearby interaction\n".format(os.path.basename(ligand_pdb)))
                f.close()
                continue
            df_receptor_coord_lig = df_receptor_coord[df_receptor_coord['atom_number'].isin(nearby_receptor_num)]
            nearby_receptor_num = df_receptor_coord_lig['atom_number'].values
            df_receptor_bond_lig = df_receptor_bond[df_receptor_bond['atom1'].isin(nearby_receptor_num) & df_receptor_bond['atom2'].isin(nearby_receptor_num)]
            
            ### form nodes features ###
            df_ligand_x_d = pd.merge(df_ligand_coord[['atom_number','x']], df_ligand_atominfo[['atom_number','f']], on = "atom_number")
            df_ligand_x_d.reset_index()
            df_receptor_x_d = pd.merge(df_receptor_coord_lig[['atom_number','x']], df_receptor_atominfo[['atom_number','f']], on = "atom_number")
            df_receptor_x_d.reset_index()
            df_receptor_x_d.index = df_receptor_x_d.index + len(df_ligand_x_d)
            
            node_pos = list(df_ligand_x_d['x'].values) + list(df_receptor_x_d['x'].values)
            node_features = list(df_ligand_x_d['f'].values) + list(df_receptor_x_d['f'].values)
            ligand_atom_index_dict = dict(zip(df_ligand_x_d['atom_number'], df_ligand_x_d.index))
            receptor_atom_index_dict = dict(zip(df_receptor_x_d['atom_number'], df_receptor_x_d.index))
            
            ### form bond features ###
            edge_data_w = list(df_ligand_bond["onehot_bond"].values) + list(df_ligand_bond["onehot_bond"].values) + list(df_receptor_bond_lig["onehot_bond"].values) + list(df_receptor_bond_lig["onehot_bond"].values)
            ligandedge_atom_number = [(ligand_atom_index_dict[atom1],ligand_atom_index_dict[atom2]) for atom1, atom2 in df_ligand_bond[['atom1', 'atom2']].values]
            receptoredge_atom_number = [(receptor_atom_index_dict[atom1],receptor_atom_index_dict[atom2]) for atom1, atom2 in df_receptor_bond_lig[['atom1', 'atom2']].values]
            ligandedge_atom_number_bi = ligandedge_atom_number + [(b,a) for a,b in ligandedge_atom_number]
            edge_index = ligandedge_atom_number_bi + receptoredge_atom_number + [(b,a) for a,b in receptoredge_atom_number]
            
            full_ligand_pairs = [(ligand_atom_index_dict[a], ligand_atom_index_dict[b]) for a,b in neighborbond_list]
            full_ligand_pairs = set(full_ligand_pairs) - set(ligandedge_atom_number_bi)
            edge_index.extend(list(full_ligand_pairs))
            edge_data_w.extend([list(np.zeros(len(bond_types)).astype(int))] * len(full_ligand_pairs))
            
            dist_matrix = np.linalg.norm(np.vstack(df_ligand_coord['x'].values)  [:, None, :] - np.vstack(df_receptor_coord_lig['x'].values)[None, :, :], axis=2)
            lig_idx_list, rec_idx_list = np.where(dist_matrix < 5)
            lig_atom_numbers = df_ligand_coord.iloc[lig_idx_list]['atom_number'].values
            rec_atom_numbers = df_receptor_coord_lig.iloc[rec_idx_list]['atom_number'].values
            for lig_idx, rec_idx in np.stack([lig_atom_numbers, rec_atom_numbers], axis=1):
                edge_data_w.append(list(inter_bond))
                edge_index.append((ligand_atom_index_dict[lig_idx], receptor_atom_index_dict[rec_idx]))
                edge_data_w.append(list(inter_bond))
                edge_index.append((receptor_atom_index_dict[rec_idx], ligand_atom_index_dict[lig_idx]))
            
            fileout = os.path.join(output_folder, os.path.basename(ligand_pdb).replace(".pdb", ".graph"))
            pyg_data = Data(x=torch.tensor(np.array(node_features), dtype=torch.float),
                            pos=torch.tensor(np.array(node_pos), dtype=torch.float),
                            edge_index=torch.tensor(np.array(edge_index), dtype=torch.int64).t().contiguous(),
                            edge_attr=torch.tensor(np.array(edge_data_w), dtype=torch.int),n=len(df_ligand_atominfo),
                            dir = fileout) # adj = torch.tensor(adj, dtype=torch.float), edges=torch.tensor(edges, dtype=torch.float), y=rmsd)
            pickle.dump(pyg_data, open(fileout, 'wb'))
    except Exception as e:
        with open(error_log, "a") as f:
            f.write("Error: {},{},{}\n".format(os.path.basename(receptor_mol2), os.path.basename(ligand_pdb), e))
        f.close()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="python generate_distance_feature.py -inp rdock_allresult_all.csv -cutoff 10 -output_folder 3_distance_rdock_docking -ncpus 8")
    parser.add_argument("-receptor", type=str, help="receptor mol2 file")
    parser.add_argument("-ligand", type=str, help = "folder containing .pdbqt, .mol2 and .pdb format of ligand", default="ligand_structure")
    parser.add_argument("-output_folder", type=str, help = "output folder containing .graph files", default="graph_egnn")
    parser.add_argument("-error", type=str, help = "error log file", default="3_error_pdb2graph.log")
    parser.add_argument("-ncpus", type=int, help = "no. of processors", default=32)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_folder, exist_ok = True)
    torch.set_num_threads(2)
    error_log = args.error
    if os.path.exists(error_log):
        os.remove(error_log)
    receptor_mol2 = args.receptor
    inputs = list(map(lambda x: os.path.basename(os.path.dirname(x)), glob(args.ligand + "/*/")))
    pool = Pool(min(args.ncpus,len(inputs)))
    pool.map(mol2graph, inputs)


