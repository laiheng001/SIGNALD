import os
from pymol import cmd
import pandas as pd
import shutil, argparse
from multiprocessing import Pool
from glob import glob

def convert_rec(receptor_mol2):
    receptor_basename = os.path.basename(receptor_mol2)
    receptor_pdb = receptor_basename.replace('.mol2', '.pdb')
    cmd.reinitialize()
    cmd.load(receptor_mol2)
    cmd.alter("all","chain='' ")
    cmd.save(f"{receptor_pdb}")
    cmd.save(f"{receptor_basename}")
    os.system("{} -r {}".format(prepare_receptor, receptor_pdb))
    os.system("grep 'ATOM\|HETATM' {basename}qt > {basename}qt2 && mv {basename}qt2 {basename}qt".format(basename = receptor_pdb))
    return

def convert_lig(lig):
    output_lig_dir = os.path.join(args.output, lig)
    ligand_pdbqt = os.path.join(output_lig_dir, lig + ".pdbqt")
    if not os.path.exists(ligand_pdbqt):
        os.makedirs(output_lig_dir, exist_ok= True)
        ligand_list = glob(os.path.join(args.docking, lig) + f"/{lig}_out*.mol2")
        for ligand_mol2 in ligand_list:
            output_ligand_mol2 = os.path.join(output_lig_dir, os.path.basename(ligand_mol2))
            shutil.copy(ligand_mol2, output_ligand_mol2)
            cmd.reinitialize()
            cmd.load(output_ligand_mol2)
            cmd.save(output_ligand_mol2.replace(".mol2", ".pdb"))
            cmd.save(output_ligand_mol2)
        output_ligand_mol2 = os.path.join(output_lig_dir, os.path.basename(ligand_list[0]))
        ligand_pdb = output_ligand_mol2.split("_out")[0] + ".pdb"
        ligand_mol2 = ligand_pdb.replace(".pdb", ".mol2")
        ligand_pdbqt = ligand_pdb + "qt"
        cmd.save(ligand_pdb)
        cmd.save(ligand_mol2)
        os.system(f"{obabel} -ipdb {ligand_pdb} -opdbqt -O {ligand_pdbqt}")
        os.system(f"grep 'ATOM\|HETATM' {ligand_pdbqt} > {ligand_pdbqt}2 && mv {ligand_pdbqt}2 {ligand_pdbqt}")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="python generate_distance_feature.py -inp rdock_allresult_all.csv -cutoff 10 -output_folder 3_distance_rdock_docking -ncpus 8")
    parser.add_argument("-receptor", type=str, help="receptor mol2")
    parser.add_argument("-docking", type=str, help="ligand_docking_folder", default = "../docking")
    parser.add_argument("-output", type=str, help="output folder for converted ligand files", default="ligand_structure")
    parser.add_argument("-ncpus", type=int, help = "no. of processors", default=32)
    args = parser.parse_args()

    # SOFTWARE path #
    obabel = "obabel"
    prepare_receptor = "prepare_receptor"
    
    os.system("cp {} ./".format(args.receptor))
    lig_list = list(map(lambda x: os.path.basename(os.path.dirname(x)), glob(args.docking + "/*/")))
    os.makedirs(args.output, exist_ok= True)
    
    convert_rec(os.path.basename(args.receptor))
    pool = Pool(args.ncpus)
    pool.map(convert_lig, lig_list)
