from rdkit import Chem
# from rdkit.Chem import rdchem
import numpy as np
from biopandas.pdb import PandasPdb
from biopandas.mol2 import PandasMol2
import pandas as pd
import os, argparse
from multiprocessing import Pool
import json
from glob import glob

"""
Version 1: Interaction only

Only consider atoms within cutoff range of ligand

### Graph1 ### basic graph, not included
Node: [one-hot residue type, one-hot element type]
Edge: [intermolecular Distance]

### GRAPH2 ### Interaction as node
Node: [one-hot residue type, one-hot element type] of ligand & [one-hot residue type, one-hot element type] of receptor & [intermolecular Distance]
Edge: [Angle] (Connect if come from same atom)
"""

def load_dict(json_file, alt_dict = {}):
    try:
        with open(json_file, 'r') as f:
            output_dictionary = json.load(f)  
            f.close()
    except json.decoder.JSONDecodeError:
        print("Error parsing dictionary in json file.")
        output_dictionary=alt_dict
    except FileNotFoundError:
        print("json file, {} not found.".format(json_file))
        output_dictionary=alt_dict
    return output_dictionary

with open("../config/residue_allNA.json", 'r') as f:
    residue_info = json.load(f)  
    f.close()

metal_elem = {'AG', 'AU', 'BA', 'CA', 'CD', 'CO', 'CR', 'CS', 'CU', 'FE', 'K', 'HG', 'MG', 'MN', 'NA', 'NI', 'PT', 'RB', 'RH', 'RU', 'SR', 'SE', 'TL', 'ZN'}
element_type = ['C', 'N', 'O', 'P', 'S', 'F', 'CL', 'BR', 'I', 'SE', 'Metal']
residue_type = ['N1', 'N2', 'backbone', 'purine', 'pyrimidine', 'OTH', "Ligand"]

# one-hot encoding
nelec_type = [4,5,6,7]
valence_type = [1,2,3,4]
degree_type = [1,2,3,4,5,6]
bond_types = ['1', '2', '3', 'ar', 'inter']
ligand_res_onehot = np.zeros(len(residue_type)).astype(int)
ligand_res_onehot[residue_type.index("Ligand")] = 1
ligand_res_onehot = list(ligand_res_onehot)
metal_element_onehot = np.zeros(len(element_type)).astype(int)
metal_element_onehot[element_type.index("Metal")] = 1
metal_element_onehot = list(metal_element_onehot)

def get_rdkit_info(pdb_file):
    mol = Chem.MolFromPDBFile(pdb_file, sanitize=False)
    info_list = []
    for atom in mol.GetAtoms():
        serialnumber = atom.GetPDBResidueInfo().GetSerialNumber()
        degree = atom.GetDegree()
        mass = atom.GetMass()
        inRing = 1 if atom.IsInRing() else 0
        symbol = atom.GetSymbol()
        vdw_radius = Chem.GetPeriodicTable().GetRvdw(symbol)
        covalent_radius = Chem.GetPeriodicTable().GetRcovalent(symbol)
        bond_radius = Chem.GetPeriodicTable().GetRb0(symbol)
        valence = Chem.GetPeriodicTable().GetDefaultValence(symbol)
        nelec = Chem.GetPeriodicTable().GetNOuterElecs(symbol)
        info_list.append([serialnumber, symbol.upper(), degree, mass, inRing, vdw_radius, covalent_radius, bond_radius, valence, nelec])
    df = pd.DataFrame(info_list, columns = ["atom_number", "symbol", 'degree', 'mass', 'inRing', 'vdw_radius', 'covalent_radius', 'bond_radius', 'valence', 'nelec'])
    return df

def modify_res(residue_name, atom_name):
    # Generalize residue type, atom_name, and mol2 type
    # try:
    # "Counter ion"
    if (residue_name in residue_info.keys()) == False:
        return "Metal"
    else:
        residue_group = residue_info[residue_name]
        if atom_name in residue_group["backbone"]:
            return "backbone"
        elif atom_name in residue_group["purine"]:
            return "purine"
        elif atom_name in residue_group["pyrimidine"]:
            return "pyrimidine"
        elif atom_name in residue_group["OTH"]:
            return "OTH"
        else:
            return "Metal"

def bond_parse(filename):
    """
    Generate df with columns ['bond_id', 'atom1', 'atom2', 'bond_type'] from .mol2 file
    """
    # filename = mol2_file
    with open(filename, 'r') as f:
        f_text = f.read()
        f.close()
    bond_start = f_text.find('@<TRIPOS>BOND')
    f_text = f_text[bond_start:].replace('@<TRIPOS>BOND\n','')
    bond_end = f_text.find('@')
    if (bond_end == -1):
        bond_end = len(f_text)
    bonds = list(filter(None, [line.split() for line in f_text[:bond_end].split("\n")]))
    df_bonds = pd.DataFrame(bonds, columns=['bond_id', 'atom1', 'atom2', 'bond_type'])
    df_bonds.dropna(subset=['atom1', 'atom2'], inplace=True)
    df_bonds[['bond_id', 'atom1', 'atom2']] = df_bonds[['bond_id', 'atom1', 'atom2']].apply(pd.to_numeric)
    df_bonds.set_index(['bond_id'], inplace=True)
    return df_bonds

def pdbtopandas(pdb_file):
    pdb_pandas = PandasPdb().read_pdb(pdb_file)
    df_atominfo = pd.concat([pdb_pandas.df["ATOM"], pdb_pandas.df["HETATM"]], ignore_index= True)
    df_atominfo['element_symbol'] = df_atominfo['element_symbol'].str.upper()
    df_atominfo_noH = df_atominfo[~df_atominfo['element_symbol'].isin(('H','D'))]
    return df_atominfo_noH

def mol2topandas(mol2_file):
    df_mol2 = PandasMol2().read_mol2(mol2_file).df
    df_mol2 = df_mol2.rename(columns={"atom_id": "atom_number", "atom_type": "mol2atom", "charge": "partialChrg"})
    return df_mol2

def pdbqt_parse(filename):
    df=pd.read_fwf(filename, colspecs = [(0, 6), (6, 11), (12, 16), (16,17), (17, 21), (21, 22), (22, 26), (26, 27),
                                         (30, 38), (38, 46), (46, 54), (54, 60), (60, 66), (66, 76), (77, 80)],
                   header = None, keep_default_na=False)
    df.columns = ["Record name", "atom_id", "atom_name", "Character", "residue_name", "chain", "residue_id", "AChar",
                  'x_coord','y_coord','z_coord',"occupancy", "tempFactor", "partialChrg", "pdbqtatom"]
    df = df[['x_coord','y_coord','z_coord','partialChrg']]
    # df[['x_coord','y_coord','z_coord',"occupancy", "tempFactor", "partialChrg"]] = df[['x_coord','y_coord','z_coord',"occupancy", "tempFactor", "partialChrg"]].astype("float")
    return df

def get_one_hot(column_values, reference):
    one_hot = np.zeros(len(reference)).astype(int)
    try:
        one_hot[reference.index(column_values)] = 1
        return list(one_hot)
    except ValueError:
        return list(one_hot)

def mol2df_coord(pdb_file, mol2_file):
    """
    Read .mol2 and .pdb file, generate pandas df (remove h) and bond df coordinates
    """
    # pdb_file, mol2_file, pdbqt_file = ligand_pdb, ligand_mol2, ligand_pdbqt
    df_pdb_noH = pdbtopandas(pdb_file)
    df_pdb_noH["x"] = df_pdb_noH.apply(lambda row: np.array([row['x_coord'], row['y_coord'], row['z_coord']]), axis = 1)
    df_mol2 = mol2topandas(mol2_file)
    df_bond = bond_parse(mol2_file)
    h_index = df_mol2[df_mol2['mol2atom'].isin(('H','D'))]["atom_number"].values
    df_bond = df_bond[~(df_bond['atom1'].isin(h_index) | df_bond['atom2'].isin(h_index))]
    df_bond = df_bond.merge(df_pdb_noH[["atom_number", "x"]].rename(columns={"atom_number": "atom1", "x": "atom1_coord"}), on="atom1", how="left")
    df_bond = df_bond.merge(df_pdb_noH[["atom_number", "x"]].rename(columns={"atom_number": "atom2", "x": "atom2_coord"}), on="atom2", how="left")
    # df_bond["d"] = df_bond.apply(lambda row: row["atom2_coord"] - row["atom1_coord"], axis=1)
    # df_bond["r"] = df_bond["d"].apply(lambda v: np.linalg.norm(v))
    df_coord_noH = pd.merge(df_pdb_noH, df_mol2[['atom_number',"mol2atom"]], on = ['atom_number'])[['atom_number', "mol2atom", "residue_name", 'x_coord','y_coord','z_coord', 'x']]
    return df_coord_noH.reset_index(), df_bond.reset_index()

def mol2df_atominfo(pdb_file, mol2_file, pdbqt_file = None):
    """
    Read .mol2 file, generate pandas df (remove h) atom_number, rdkit + charge information
    """
    # pdb_file, mol2_file, pdbqt_file = ligand_pdb, ligand_mol2, ligand_pdbqt
    df_pdb_noH = pdbtopandas(pdb_file)
    
    if pdbqt_file != None:
        df_pdbqt = pdbqt_parse(pdbqt_file)
        df_pdbqt_atomid = pd.merge(df_pdb_noH[["atom_number",'x_coord','y_coord','z_coord']], df_pdbqt, on = ['x_coord','y_coord','z_coord'])
        df_mol2 = df_pdbqt_atomid[["atom_number", "partialChrg"]]
    else:
        df_mol2 = mol2topandas(mol2_file)
    
    df_atominfo = pd.merge(df_mol2[['atom_number', "partialChrg"]], df_pdb_noH[['atom_number', "atom_name", "residue_name", 'element_symbol']], on = ['atom_number'])
    df_rdkit = get_rdkit_info(pdb_file)
    df_atominfo = pd.merge(df_atominfo, df_rdkit, on = ['atom_number'])
    df_atominfo["element_symbol"] = df_atominfo.apply(lambda x: "Metal" if x['element_symbol'] in metal_elem else x['element_symbol'], axis = 1)
    df_atominfo["onehot_element"] = df_atominfo.apply(lambda x: get_one_hot(x["element_symbol"], element_type), axis = 1)
    df_atominfo["values_rdkit"] = df_atominfo.apply(lambda x: get_one_hot(x["nelec"], nelec_type) +
                                                              get_one_hot(x["valence"], valence_type) + 
                                                              get_one_hot(x["degree"], degree_type) +
                                                              [x["partialChrg"],x["mass"],x["inRing"], x["vdw_radius"],x["covalent_radius"], x['bond_radius']]
                                                              , axis = 1)
    return df_atominfo.reset_index()

# receptor_mol2 = '/home/laiheng/RmsdXNA/3_structure/na_structure/6T3R_model1/6T3R_model1_na.mol2'
def receptor2csv(receptor_mol2):
    receptor_pdb = receptor_mol2.replace(".mol2", ".pdb")
    try:
        if os.path.exists(receptor_pdb.replace(".pdb", "_bond.parquet")):
            return
        else:
            df_receptor_coord, df_receptor_bond = mol2df_coord(receptor_pdb, receptor_mol2)
            df_receptor_atominfo = mol2df_atominfo(receptor_pdb, receptor_mol2, receptor_pdb + "qt")
            
            df_receptor_atominfo["residue2"] = df_receptor_atominfo.apply(lambda x: modify_res(x['residue_name'], x['atom_name']), axis = 1)
            df_receptor_atominfo["N1"] = 0
            df_receptor_atominfo["N2"] = 0
            
            residue_to_indices = df_receptor_atominfo.groupby("residue_name").groups
            for residue, indices in residue_to_indices.items():
                residue_dict = residue_info.get(residue)
                if not residue_dict:
                    continue
                if residue_dict.get("O3"):
                    df_receptor_atominfo.loc[indices, "N1"] = 1
                if residue_dict.get("O2"):
                    df_receptor_atominfo.loc[indices, "N2"] = 1
            df_receptor_aromatic = df_receptor_atominfo[df_receptor_atominfo["residue2"].isin(["purine", "pyrimidine"])]
            aromatic_atom_id = df_receptor_aromatic["atom_number"].values
            df_receptor_bond.loc[(df_receptor_bond['atom1'].isin(aromatic_atom_id)) & (df_receptor_bond['atom2'].isin(aromatic_atom_id)), 'bond_type'] = "ar"
            df_receptor_bond["onehot_bond"] = df_receptor_bond.apply(lambda x: get_one_hot(x["bond_type"], bond_types), axis = 1)
            df_receptor_atominfo["aromaticity"] = df_receptor_atominfo.apply(lambda x: 1 if x["atom_number"] in aromatic_atom_id else 0, axis = 1)
            df_receptor_atominfo["onehot_residue"] = df_receptor_atominfo.apply(lambda x: [x["N1"], x["N2"]] + get_one_hot(x["residue2"], residue_type)[2:], axis = 1)
            # Remove problematic receptor atoms
            problem_residue = df_receptor_atominfo[df_receptor_atominfo["onehot_residue"].apply(lambda x: x == list(np.zeros(len(residue_type)).astype(int)))]
            problem_element = df_receptor_atominfo[df_receptor_atominfo["element_symbol"].apply(lambda x: x in [list(np.zeros(len(element_type)).astype(int)), metal_element_onehot])]
            if len(problem_residue) > 0:
                df_receptor_atominfo = df_receptor_atominfo[~df_receptor_atominfo["onehot_residue"].apply(lambda x: x == list(np.zeros(len(residue_type)).astype(int)))]
                with open(error_log, "a") as f:
                    f.write("{} Error: Unknown residues {}\n".format(os.path.basename(receptor_mol2), problem_residue["residue_name"].unique()))
                f.close()
            if len(problem_element) > 0:
                df_receptor_atominfo = df_receptor_atominfo[~df_receptor_atominfo["element_symbol"].apply(lambda x: x in [list(np.zeros(len(element_type)).astype(int)), metal_element_onehot])]
                with open(error_log, "a") as f:
                    f.write("{} Error: Unknown element {}\n".format(os.path.basename(receptor_mol2), problem_element["element_symbol"].unique()))
                f.close()
            problem_bond = [bond for bond in df_receptor_bond["bond_type"].unique() if bond not in bond_types]
            if len(problem_bond) > 0:
                df_receptor_bond = df_receptor_bond[df_receptor_bond["bond_type"].isin(bond_types)]
                with open(error_log, "a") as f:
                    f.write("{} Error: Unknown bond_type {}\n".format(os.path.basename(receptor_mol2), problem_bond))
                f.close()
            
            df_receptor_atominfo.to_parquet(receptor_pdb.replace(".pdb", "_atominfo.parquet"), index= False)
            df_receptor_coord.to_parquet(receptor_pdb.replace(".pdb", "_coord.parquet"), index= False)
            df_receptor_bond.to_parquet(receptor_pdb.replace(".pdb", "_bond.parquet"), index= False)
                 
    except Exception as e:
        with open(error_log, "a") as f:
            f.write("{} Error: {}\n".format(receptor_mol2, e))
            f.close()

def ligand2csv(lig):
    try:
        # if not os.path.exists(os.path.join(ligand_folder, lig, lig + "_bond.parquet")):
        ligand0_pdb = os.path.join(ligand_folder, lig, lig + ".pdb")
        ligand0_mol2 = ligand0_pdb.replace(".pdb", ".mol2")
        ligand0_pdbqt = os.path.join(ligand_folder, lig, lig + ".pdbqt")
        df_ligand_atominfo = mol2df_atominfo(ligand0_pdb, ligand0_mol2, ligand0_pdbqt)
        _, df_ligand_bond = mol2df_coord(ligand0_pdb, ligand0_mol2)
        df_aromatic_atoms = df_ligand_bond[df_ligand_bond["bond_type"] == "ar"]
        aromatic_atoms = set(list(df_aromatic_atoms['atom1'].unique()) + list(df_aromatic_atoms['atom2'].unique()))
        df_ligand_bond["onehot_bond"] = df_ligand_bond.apply(lambda x: get_one_hot(x["bond_type"], bond_types), axis = 1)
        df_ligand_atominfo["aromaticity"] = df_ligand_atominfo.apply(lambda x: 1 if x["atom_number"] in aromatic_atoms else 0, axis = 1)
        df_ligand_atominfo["onehot_residue"] = [ligand_res_onehot] * len(df_ligand_atominfo)
        
        problem_element = df_ligand_atominfo[df_ligand_atominfo["element_symbol"].apply(lambda x: x == list(np.zeros(len(element_type)).astype(int)))]
        if len(problem_element) > 0:
            df_ligand_atominfo = df_ligand_atominfo[~df_ligand_atominfo["element_symbol"].apply(lambda x: x == list(np.zeros(len(element_type)).astype(int)))]
            with open(error_log, "a") as f:
                f.write("{} Error: Unknown element {}\n".format(os.path.basename(ligand0_pdb), problem_element["element_symbol"].unique()))
            f.close()
        df_ligand_atominfo.to_parquet(os.path.join(ligand_folder, lig, lig + "_atominfo.parquet"), index= False)
        df_ligand_bond.to_parquet(os.path.join(ligand_folder, lig, lig + "_bond.parquet"), index= False)
        
        for ligand_pdb in glob(os.path.join(ligand_folder, lig) + "/*.pdb"):
            if os.path.exists(ligand_pdb.replace(".pdb", "_coord.parquet")):
                continue
            ligand_mol2 = ligand_pdb.replace(".pdb", ".mol2")
            df_ligand_coord, _ = mol2df_coord(ligand_pdb, ligand_mol2)
            df_ligand_coord.to_parquet(ligand_pdb.replace(".pdb", "_coord.parquet"), index= False)
    except Exception as e:
        with open(error_log, "a") as f:
            f.write("{} Error: {}\n".format(lig, e))
            f.close()
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="python generate_distance_feature.py -inp rdock_allresult_all.csv -cutoff 10 -output_folder 3_distance_rdock_docking -ncpus 8")
    parser.add_argument("-receptor", type=str, help="receptor mol2")
    parser.add_argument("-ligand", type=str, help = "folder containing .pdbqt, .mol2 and .pdb format of ligand", default="ligand_structure")
    parser.add_argument("-error", type=str, help = "error log file", default="2_error_getcsv.log")
    parser.add_argument("-ncpus", type=int, help = "no. of processors", default=32)
    args = parser.parse_args()
    
    ### load input.dat file ###
    """
    For pdb_splitchain, na.pdb lig.pdb
    For docking, na.pdb lig.mol2
    """
    ligand_folder = args.ligand
    error_log = args.error

    pool = Pool(args.ncpus)
    receptor2csv(args.receptor)
    pool.map(ligand2csv, list(map(lambda x: os.path.basename(os.path.dirname(x)), glob(f"{ligand_folder}/*/"))))

