# SIGNALD

> **SIGNALD** is a generalizable scoring function for pose identification and evaluation across RNA and DNA (including modified Nucleic-Acid residues) complexes.

---
> [!Note]
> The trained model can be obtained upon request

## 🧩 Installation

### 1. Create and activate environment

```bash
conda create -n signald python=3.10
conda activate signald
```

### 2. Install dependencies
```bash
conda install conda-forge::plant
conda install hcc::adfr-suite
conda install schrodinger::pymol
conda install -c conda-forge rdkit
```

### 3. Install PyTorch and PyTorch Geometric
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install torch_geometric
```

## 🚀 Usage

### 1. Clone repository

```
git clone https://github.com/laiheng001/SIGNALD.git
cd SIGNALD
```

### 2. Prepare input files
- Copy your receptor file to the main directory:
- Place your ligand library (in .sdf format) inside the `ligand/` folder:

### Step 1: Perform PLANTS Docking
```
python3 1.1_run_plants.py -receptor receptor.mol2 -center "(x, y, z)
```

### Step 2: Process and Score Complexes
```
cd SIGNALD
python3 1_convert.py -receptor receptor.mol2
python3 2_getcsv.py -receptor receptor.mol2
python3 3_pdb2graph.py -receptor receptor.mol2
python3 4_graph_getscore.py

```

