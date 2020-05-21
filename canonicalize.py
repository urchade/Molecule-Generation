from rdkit import Chem
from rdkit.Chem.rdmolfiles import MolToSmiles


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    cano_smiles = MolToSmiles(mol)
    return cano_smiles
