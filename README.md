###'DataPreprocessing.py' < br / >
This script handles the preprocessing of molecular data. It processes molecular structures and extracts essential information, outputting a dictionary in the following format: < br / >
'''data_dict = { < br / >
    "ligand_smi": c1ccccc1, < br / >
    "receptor_seq": RDEVALPWW, < br / >
    "residue_index": a list indices of protein amino acid residue that the distance from the ligand is less than 8 A. < br / >
}''' < br / >


###'predict.py' < br / >
This script is an independent reproduction of the research paper "Protein Language Model-Powered 3D Ligand Binding Site Prediction from Protein Sequence." Since the original code is not publicly available, this implementation was developed from scratch based on the methods described in the paper.
