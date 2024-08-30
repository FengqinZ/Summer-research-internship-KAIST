import torch
import esm
from rdkit import Chem
from rdkit.Chem import AllChem
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import softmax
import numpy as np
from Physics-aware-Multiplex-GNN.models import PAMNet, Config
from TankBind.tankbind.model import TriangleProteinToCompound, TriangleSelfAttentionRowWise, Transition
from torch_geometric.utils import to_dense_batch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

def load_esm_model(model_path):
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
    batch_converter = alphabet.get_batch_converter()
    return esm_model, batch_converter

#Protein sequence encoder
def process_protein_sequence(protein_sequence, esm_model, batch_converter, device):
    data = [("protein", protein_sequence)]
    _, _, batch_tokens = batch_converter(data)
    
    batch_tokens = batch_tokens.to(device)
    esm_model = esm_model.to(device)

    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
    
    token_representations = results["representations"][33]
    contact_map = results["contacts"]
    
    return token_representations, contact_map

#Ligand encoder
def smiles_to_mol_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.RemoveHs(mol)  
    AllChem.Compute2DCoords(mol)

    # Atom features (e.g., atom type, charge)
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([atom.GetAtomicNum()])

    # Bond features (edges in the graph)
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i]) 

    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)

def generate_3d_conformer_and_distance_map(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)

    conf = mol.GetConformer()
    positions = conf.GetPositions()

    distance_map = np.linalg.norm(positions[:, np.newaxis] - positions, axis=2)
    distance_map = torch.tensor(distance_map, dtype=torch.float)
    
    return distance_map

def get_atom_embeddings_from_smiles(smiles, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PAMNet(config).to(device)
    model.eval()

    ligand_graph = smiles_to_mol_graph(smiles).to(device) 

    with torch.no_grad():  
        atom_embeddings = model(ligand_graph)
    
    return atom_embeddings


def compute_interaction_embedding(hp, hl):
    # Compute interaction embedding z
    z = torch.einsum('ij,kj->ikj', hp, hl)  # Element-wise product across the embedding dimension
    return z

class TrigonometryModule(nn.Module):
    def __init__(self, embedding_dim, c=128, n_heads=4):
        super(TrigonometryModule, self).__init__()
        self.embedding_dim = embedding_dim
        self.c = c
        self.layernorm = nn.LayerNorm(embedding_dim)
        self.protein_to_compound = TriangleProteinToCompound(embedding_channels=embedding_dim, c=c)
        self.self_attention = TriangleSelfAttentionRowWise(embedding_channels=embedding_dim, c=c, num_attention_heads=n_heads)
        self.transition = Transition(embedding_channels=embedding_dim, n=4)
        self.dropout = nn.Dropout(0.25)

    def forward(self, z, protein_pair, compound_pair, z_mask):
        z = self.layernorm(z)
        for i in range(1):  
            z = z + self.dropout(self.protein_to_compound(z, protein_pair, compound_pair, z_mask.unsqueeze(-1)))
            z = z + self.dropout(self.self_attention(z, z_mask))
            z = self.transition(z)
        return z
    
def compute_interaction_embedding(hp, hl, Cp, Dl):
    z = torch.einsum('ij,kj->ikj', hp, hl)  

    protein_out_batched, protein_out_mask = to_dense_batch(hp)
    compound_out_batched, compound_out_mask = to_dense_batch(hl)
    z_mask = torch.einsum("bi,bj->bij", protein_out_mask.float(), compound_out_mask.float())

    trigonometry_module = TrigonometryModule(embedding_dim=z.shape[-1], c=128, n_heads=4)
    
    z_prime = trigonometry_module(z, Cp, Dl, z_mask)
    
    return z_prime

class PoolingModule(nn.Module):
    def __init__(self, embedding_dim):
        super(PoolingModule, self).__init__()
        self.linear = nn.Linear(embedding_dim, 1)  # Reducing dimension to 1

    def forward(self, z, z_prime):
        # Step 1: Sum the updated interaction embedding z' with the original z
        z_combined = z + z_prime  

        # Step 2: Apply a linear transformation to reduce the dimension to 1
        z_final = self.linear(z_combined).squeeze(-1)  # Shape: (n_p, n_l)

        # Step 3: Mean pooling over the ligand dimension (axis=1)
        s = torch.mean(z_final, dim=1)  # Shape: (n_p,)

        return s, z_final

def main_pooling_module(z, z_prime):
    pooling_module = PoolingModule(embedding_dim=z.shape[-1])
    s, z_final = pooling_module(z, z_prime)
    
    return s, z_final

def normalize_scores(scores):
    """Normalize the scores to the range [0, 1]."""
    min_score = scores.min()
    max_score = scores.max()
    return (scores - min_score) / (max_score - min_score)

def filter_residues(scores, threshold):
    """Filter residues with scores less than the threshold."""
    return scores >= threshold

def single_linkage_clustering(contact_map, filtered_indices, distance_threshold=1.0):
    # Extract the relevant part of the contact map for the filtered residues
    filtered_contact_map = contact_map[filtered_indices][:, filtered_indices]

    # Convert the contact map to a distance matrix (1 - contact map)
    distance_matrix = 1.0 - filtered_contact_map

    # Perform single linkage clustering
    condensed_distance_matrix = squareform(distance_matrix, checks=False)
    Z = linkage(condensed_distance_matrix, method='single')
    
    # Form clusters based on the distance threshold
    clusters = fcluster(Z, t=distance_threshold, criterion='distance')

    return clusters

def rank_clusters(clusters, filtered_scores):
    """Rank clusters based on the squared sum of the associated residue scores."""
    unique_clusters = np.unique(clusters)
    cluster_scores = []

    for cluster_id in unique_clusters:
        cluster_residues = np.where(clusters == cluster_id)[0]
        cluster_score = np.sum(filtered_scores[cluster_residues] ** 2)
        cluster_scores.append((cluster_id, cluster_score))

    # Sort clusters by their scores in descending order
    ranked_clusters = sorted(cluster_scores, key=lambda x: x[1], reverse=True)

    return ranked_clusters

def clustering_and_ranking(scores, contact_map, v=0.5, distance_threshold=1.0):
    # Step 1: Normalize scores to [0, 1]
    normalized_scores = normalize_scores(scores)

    # Step 2: Filter residues by threshold
    filtered_indices = torch.where(normalized_scores >= v)[0]
    filtered_scores = normalized_scores[filtered_indices]

    # Step 3: Perform clustering using single linkage algorithm
    clusters = single_linkage_clustering(contact_map, filtered_indices, distance_threshold)

    # Step 4: Rank clusters based on the squared sum of residue scores
    ranked_clusters = rank_clusters(clusters, filtered_scores)

    # Step 5: Remove clusters with fewer than 3 residues
    final_clusters = [(cluster_id, score) for cluster_id, score in ranked_clusters if np.sum(clusters == cluster_id) > 3]

    return final_clusters, filtered_indices, clusters

def predict_binding_site(data_dict, esm_model_path, pamnet_config):
    # Extract data from data_dict
    ligand_smi = data_dict["ligand_smi"]
    receptor_seq = data_dict["receptor_seq"]
    residue_index = data_dict["residue_index"]  # Indices of protein residues close to the ligand
    
    # Step 1: Load and process the ESM model
    esm_model, batch_converter = load_esm_model(esm_model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Step 2: Process the protein sequence
    hp, Cp = process_protein_sequence(receptor_seq, esm_model, batch_converter, device)
    
    # Step 3: Process the ligand
    hl = get_atom_embeddings_from_smiles(ligand_smi, pamnet_config)
    Dl = generate_3d_conformer_and_distance_map(ligand_smi).to(device)
    
    # Step 4: Compute the interaction embedding
    z = compute_interaction_embedding(hp, hl, Cp, Dl)
    
    # Step 5: Apply the pooling module to compute scores for each residue
    s, z_final = main_pooling_module(z, z)
    
    # Step 6: Filter the residue scores based on the provided residue indices
    filtered_scores = s[residue_index]
    
    # Step 7: Cluster and rank the residues to predict binding sites
    final_clusters, filtered_indices, clusters = clustering_and_ranking(filtered_scores, Cp[residue_index][:, residue_index])
    
    # Step 8: Map the filtered indices back to the original residue indices
    predicted_binding_sites = [residue_index[idx] for cluster_id, score in final_clusters for idx in np.where(clusters == cluster_id)[0]]
    
    return predicted_binding_sites

# Example usage
data_dict = {
    "ligand_smi": "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CSCC[C@H]([NH3+])C(=O)[O-])[C@@H](O)[C@H]1O",
    "receptor_seq": "XXXXXXXXXXXXXXXXXXXXXXXXMIEIKDKQLTGLRFIDLFAGLGGFRLALESCGAECVYSNEWDKYAQEVYEMNFGEKPEGDITQVNEKTIPDHDILCAGFPCQAFSISGKQKGFEDSRGTLFFDIARIVREKKPKVVFMENVKNFASHDNGNTLEVVKNTMNELDYSFHAKVLNALDYGIPQKRERIYMICFRNDLNIQNFQFPKPFELNTFVKDLLLPDSEVEHLVIDRKDLVMTNQEIEQTTPKTVRLGIVGKGGQGERIYSTRGIAITLSAYGGGIFAKTGGYLVNGKTRKLHPRECARVMGYPDSYKVHPSTSQAYKQFGNSVVINVLQYIAYNIGSSLNFKPYXXXXXXXXXXX",
    "residue_index": [["A", 66], ["A", 124], ["A", 69], ["A", 313], ["A", 72], ["A", 81], ["A", 87], ["A", 325], ["A", 331], ["A", 84], ["A", 328], ["A", 44], ["A", 102], ["A", 41], ["A", 47], ["A", 352], ["A", 105], ["A", 50], ["A", 355], ["A", 120], ["A", 187], ["A", 65], ["A", 77], ["A", 309], ["A", 324], ["A", 83], ["A", 327], ["A", 86], ["A", 330], ["A", 101], ["A", 40], ["A", 278], ["A", 104], ["A", 43], ["A", 49], ["A", 46], ["A", 357], ["A", 64], ["A", 70], ["A", 67], ["A", 73], ["A", 18], ["A", 82], ["A", 326], ["A", 85], ["A", 143], ["A", 329], ["A", 88], ["A", 332], ["A", 277], ["A", 335], ["A", 42], ["A", 100], ["A", 106], ["A", 45], ["A", 51], ["A", 103], ["A", 48], ["A", 353], ["A", 127], ["A", 63], ["A", 121], ["A", 307]]
}

esm_model_path = "/home/fengqin/internship/LaMPSite/esm2_pretrained_model.pt"
pamnet_config = Config(dataset="PDBbind", dim=128, n_layer=3, cutoff_l=5.0, cutoff_g=10.0)


predicted_binding_sites = predict_binding_site(data_dict, esm_model_path, pamnet_config)
print(predicted_binding_sites)