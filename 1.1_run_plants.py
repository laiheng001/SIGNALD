import os
import pandas as pd
from glob import glob
from multiprocessing import Pool
from pymol import cmd
import subprocess
import shutil
import os, argparse

def parse_tuple(coord_str):
    coord_str = coord_str.strip("()[] ").replace(" ", "")
    try:
        return tuple(map(float, coord_str.split(",")))
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid coordinate format: '{coord_str}'. Expected format like (x, y, z) or [x, y, z]")
    
def parameter_text(receptor, ligand, output_dir, center_x, center_y, center_z, box_size):
    plants_config = f"""
# scoring function and search settings
scoring_function chemplp
search_speed speed1
# input
protein_file {receptor}
ligand_file {ligand}
# output
output_dir {output_dir}
# write single mol2 files (e.g. for RMSD calculation)
# write_multi_mol2 0
# binding site definition
bindingsite_center {center_x} {center_y} {center_z}
bindingsite_radius {box_size}
# cluster algorithm
cluster_structures 100
cluster_rmsd 1.0
"""
    return plants_config

def process(ligand_sdf):
    ligand = ligand_sdf.replace(".sdf", "")
    ligand_docking_folder = os.path.join(args.docking, ligand)
    mol2_ligand = ligand_sdf.replace(".sdf", ".mol2")
    dock_out = os.path.join(ligand_docking_folder, "docked_ligands.mol2")
    cmd.reinitialize()
    cmd.load(os.path.join(args.ligand, ligand_sdf))
    cmd.save(mol2_ligand)
    plants_config = parameter_text(args.receptor, mol2_ligand, ligand_docking_folder, center[0], center[1], center[2], args.radius)
    prm_text_filepath = ligand + ".txt"
    with open(prm_text_filepath, "w") as f_prm:
        f_prm.write(plants_config)
        f_prm.close()
    try:
        subprocess.run(["plants", "--mode", "screen", prm_text_filepath], check=True)
        print("Docking completed successfully.")
    except subprocess.CalledProcessError as e:
        print("Docking failed:")
        print(e.stderr)
    shutil.move(prm_text_filepath, ligand_docking_folder)
    shutil.move(mol2_ligand, ligand_docking_folder)
    
    try:
        cmd.reinitialize()
        cmd.load(dock_out)
        cmd.save(dock_out, state = 0)
        os.system("{} -imol2 {} -omol2 -O {} -m".format(obabel, dock_out, os.path.join(ligand_docking_folder, os.path.basename(mol2_ligand).replace(".mol2", "_out.mol2"))))
        df_all = pd.read_csv(os.path.join(ligand_docking_folder,"ranking.csv"), usecols=["TOTAL_SCORE"])
        df_all["pose_no"] = df_all.index + 1
        df_all["Name"] = df_all.apply(lambda x: os.path.basename(mol2_ligand).replace(".mol2", "_out{}".format(int(x["pose_no"]))), axis = 1)
        df_all.reset_index(drop=True, inplace = True)
        df_all.to_csv(os.path.join(args.docking, ligand + ".csv"), index = False)
    except Exception as e:
        with open(args.error, "a") as flog:
            flog.write("{} unable to get score, {}\n".format(ligand, e))
            flog.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="python generate_distance_feature.py -inp rdock_allresult_all.csv -cutoff 10 -output_folder 3_distance_rdock_docking -ncpus 8")
    parser.add_argument("-receptor", type=str, help="receptor mol2")
    parser.add_argument("-size", type=float, help="docking box radius", default = 15)
    parser.add_argument("-center", type=parse_tuple, help="Docking site center coordinate. E.g. (x, y, z) or [x, y, z]")
    parser.add_argument("-ligand", type=str, help = "ligand library folder containing .sdf files", default="ligand")
    parser.add_argument("-docking", type=str, help = "docking output folder", default="docking")
    parser.add_argument("-ncpus", type=int, help = "no. of processors", default=32)
    parser.add_argument("-error", type=str, help = "error log", default="1.1_error_docking.log")
    args = parser.parse_args()

    obabel = "obabel" # path to obabel
    os.makedirs(args.docking, exist_ok=True)
    center = args.center
    input_list = list(map(lambda x: os.path.basename(x).replace(".mol2", ""), glob(args.ligand + "/*.sdf")))

    pool = Pool(min(args.ncpus, len(input_list)))
    pool.map(process, input_list) 

# awk '(NR == 1) || (FNR > 1)' docking/*.csv > 1_plants_allresult.csv
