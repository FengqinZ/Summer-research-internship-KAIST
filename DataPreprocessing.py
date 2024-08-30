import os
import glob
from rdkit import Chem
from Bio.PDB import PDBParser, NeighborSearch, Selection
import numpy as np
from Bio.SeqUtils import seq1
import subprocess
import json


def mol2_to_smiles(mol2_path):
    mol = Chem.MolFromMol2File(mol2_path)
    if mol:
        smiles = Chem.MolToSmiles(mol)
        return smiles
    return None

def convert_mol2_to_pdb(mol2_path, pdb_path):
    command = f"obabel {mol2_path} -O {pdb_path}"
    subprocess.run(command, shell=True, check=True)

def extract_receptor_sequence(protein_path):
    parser = PDBParser()
    structure = parser.get_structure("protein", protein_path)
    sequence = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == " ":
                    sequence += seq1(residue.resname)
    return sequence


def get_residue_indices_within_distance(ligand_path, protein_path, distance_threshold=8.0):
    parser = PDBParser()
    structure = parser.get_structure("protein", protein_path)
    
    ligand = Chem.MolFromMol2File(ligand_path)
    if ligand is None:
        return []

    # Get ligand atom coordinates
    ligand_conformer = ligand.GetConformer()
    ligand_coords = np.array([ligand_conformer.GetAtomPosition(i) for i in range(ligand.GetNumAtoms())])
    
    # Get all protein atoms, filtering out non-atom records
    protein_atoms = [atom for atom in Selection.unfold_entities(structure, 'A') if atom.element]
    ns = NeighborSearch(protein_atoms)
    
    residue_indices = set()
    for ligand_coord in ligand_coords:
        close_atoms = ns.search(ligand_coord, distance_threshold)
        for atom in close_atoms:
            residue = atom.get_parent()
            chain_id = residue.get_parent().id
            if residue.id[0] == " ":  # standard residue
                residue_indices.add((chain_id, residue.id[1]))  # Add (chain ID, residue index) to the set

    return list(residue_indices)

def preprocess_data(dataset_path):
    data_dict = {
        "ligand_smi": [],
        "receptor_seq": [],
        "residue_index": []
    }
    
    folders = glob.glob(os.path.join(dataset_path, "*"))
    print(f"Found {len(folders)} folders in dataset path: {dataset_path}")
    
    for folder in folders:
        ligand_mol2_path = os.path.join(folder, "ligand.mol2")
        protein_mol2_path = os.path.join(folder, "protein.mol2")

        protein_pdb_path = os.path.join(folder, "protein.pdb")
        
        convert_mol2_to_pdb(protein_mol2_path, protein_pdb_path)
        
        ligand_smi = mol2_to_smiles(ligand_mol2_path)
        if not ligand_smi:
            continue
        
        receptor_seq = extract_receptor_sequence(protein_pdb_path)
        residue_indices = get_residue_indices_within_distance(ligand_mol2_path, protein_pdb_path)

        data_dict["ligand_smi"].append(ligand_smi)
        data_dict["receptor_seq"].append(receptor_seq)
        data_dict["residue_index"].append(residue_indices)
    
    return data_dict

# Usage
dataset_path = "/home/fengqin/internship/LaMPSite/scPDB"
data_dict = preprocess_data(dataset_path)

processed_data_path = "/home/fengqin/internship/LaMPSite/ProcessedData.txt"
with open(processed_data_path, 'w') as f:
    json.dump(data_dict, f, indent=4)

# ligand_path = "/home/fengqin/internship/LaMPSite/scPDB/1a2n_1/ligand.mol2"
# protein_path = "/home/fengqin/internship/LaMPSite/scPDB/1a2n_1/protein.pdb"
# ligand_smi = mol2_to_smiles(ligand_path)
# receptor_seq = extract_receptor_sequence(protein_path)
# residue_index = get_residue_indices_within_distance(ligand_path, protein_path)
# print("ligand_smi:", ligand_smi)
# print("receptor_seq:", receptor_seq)
# print("residue_index:", residue_index)
# processed_data_path = "/home/fengqin/internship/LaMPSite/ProcessedData.txt"
# with open(processed_data_path, 'w') as f:
#     json.dump(residue_index, f)