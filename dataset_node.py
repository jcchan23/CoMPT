import math
import torch
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem


def one_hot_vector(val, lst, add_unknown=True):
    if add_unknown:
        vec = np.zeros(len(lst) + 1)
    else:
        vec = np.zeros(len(lst))

    vec[lst.index(val) if val in lst else -1] = 1
    return vec


def get_atom_features(atom, atom_hidden, atom_rings=None):
    # 100+1=101 dimensions
    v1 = one_hot_vector(atom.GetAtomicNum(), [i for i in range(1, 101)])

    # 5+1=6 dimensions
    v2 = one_hot_vector(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                                  Chem.rdchem.HybridizationType.SP2,
                                                  Chem.rdchem.HybridizationType.SP3,
                                                  Chem.rdchem.HybridizationType.SP3D,
                                                  Chem.rdchem.HybridizationType.SP3D2])
    # 6 dimensions
    # v3 = [0 for _ in range(6)]
    # for ring in atom_rings:
    #     if atom in ring and len(ring) <= 8:
    #         v3[len(ring) - 3] += 1

    # 8 dimensions
    v4 = [
        atom.GetTotalNumHs(includeNeighbors=True) / 8,
        atom.GetDegree() / 4,
        atom.GetFormalCharge() / 8,
        atom.GetTotalValence() / 8,
        0 if math.isnan(atom.GetDoubleProp('_GasteigerCharge')) or math.isinf(atom.GetDoubleProp('_GasteigerCharge')) else atom.GetDoubleProp('_GasteigerCharge'),
        0 if math.isnan(atom.GetDoubleProp('_GasteigerHCharge')) or math.isinf(atom.GetDoubleProp('_GasteigerHCharge')) else atom.GetDoubleProp('_GasteigerHCharge'),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing())
    ]

    # index for position encoding
    v5 = [
        atom.GetIdx() + 1       # start from 1
    ]

    attributes = np.concatenate([v1, v2, v4, v5], axis=0)

    # total for 32 dimensions
    assert len(attributes) == atom_hidden + 1
    return attributes


def get_bond_features(bond, bond_hidden, bond_rings=None):
    # 4 dimensions
    v1 = one_hot_vector(bond.GetBondType(), [Chem.rdchem.BondType.SINGLE,
                                             Chem.rdchem.BondType.DOUBLE,
                                             Chem.rdchem.BondType.TRIPLE,
                                             Chem.rdchem.BondType.AROMATIC], add_unknown=False)

    # 6 dimensions
    v2 = one_hot_vector(bond.GetStereo(), [Chem.rdchem.BondStereo.STEREOANY,
                                           Chem.rdchem.BondStereo.STEREOCIS,
                                           Chem.rdchem.BondStereo.STEREOE,
                                           Chem.rdchem.BondStereo.STEREONONE,
                                           Chem.rdchem.BondStereo.STEREOTRANS,
                                           Chem.rdchem.BondStereo.STEREOZ], add_unknown=False)

    # 6 dimensions
    # v3 = [0 for _ in range(6)]
    # for ring in bond_rings:
    #     if bond in ring and len(ring) <= 8:
    #         v3[len(ring) - 3] += 1

    # 3 dimensions
    v4 = [
        int(bond.GetIsConjugated()),
        int(bond.GetIsAromatic()),
        int(bond.IsInRing())
    ]

    # 2 dimensions for directions
    # v5 = [
    #     begin_atom.GetIdx() * 0.01,
    #     end_atom.GetIdx() * 0.01,
    # ]

    # total for 19 dimensions
    attributes = np.concatenate([v1, v2, v4])

    assert len(attributes) == bond_hidden
    return attributes


def load_data_from_mol(mol, atom_hidden, bond_hidden, max_length):

    # Set Stereochemistry
    Chem.rdmolops.AssignAtomChiralTagsFromStructure(mol)
    Chem.rdmolops.AssignStereochemistryFrom3D(mol)
    AllChem.ComputeGasteigerCharges(mol)

    # Get Node Ring
    atom_rings = mol.GetRingInfo().AtomRings()

    # Get Node features Init
    node_features = np.array([get_atom_features(atom, atom_hidden, atom_rings) for atom in mol.GetAtoms()])

    # Get Bond Ring
    bond_rings = mol.GetRingInfo().BondRings()

    # Get Bond features
    bond_features = np.zeros((mol.GetNumAtoms(), mol.GetNumAtoms(), bond_hidden))

    for bond in mol.GetBonds():
        begin_atom_idx = bond.GetBeginAtom().GetIdx()
        end_atom_idx = bond.GetEndAtom().GetIdx()
        bond_features[begin_atom_idx, end_atom_idx, :] = bond_features[end_atom_idx, begin_atom_idx, :] = \
            get_bond_features(bond, bond_hidden)

    # Get Adjacency matrix without self loop
    adjacency_matrix = Chem.rdmolops.GetDistanceMatrix(mol).astype(np.float)

    # node_features.shape    = (num_atoms, d_atom) -> (max_length, d_atom)
    # bond_features.shape    = (num_atoms, num_atoms, d_edge) -> (max_length, max_length, d_edge)
    # adjacency_matrix.shape = (num_atoms, num_atoms) -> (max_length, max_length)
    return pad_array(node_features, (max_length, node_features.shape[-1])), \
           pad_array(bond_features, (max_length, max_length, bond_features.shape[-1])), \
           pad_array(adjacency_matrix, (max_length, max_length))


def load_label_from_cs(max_length, cs):
    label = np.zeros(max_length, dtype=np.float32)
    for idx, value in cs.items():
        label[idx] = value
    label = np.reshape(label, (max_length, 1))
    return label


class Molecule:

    def __init__(self, mol, cs, atom_hidden, bond_hidden, max_length):
        self.node_features, self.bond_features, self.adjacency_matrix = \
            load_data_from_mol(mol, atom_hidden, bond_hidden, max_length)
        self.label = load_label_from_cs(max_length, cs)
        self.max_length = max_length


class MolDataSet(Dataset):

    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        if type(key) == slice:
            return MolDataSet(self.data_list[key])
        return self.data_list[key]


def pad_array(array, shape, dtype=np.float32):
    padded_array = np.zeros(shape, dtype=dtype)
    if len(shape) == 2:
        padded_array[:array.shape[0], :array.shape[1]] = array
    elif len(shape) == 3:
        padded_array[:array.shape[0], :array.shape[1], :] = array
    return padded_array


def construct_dataset(mol_list, cs_list, atom_hidden, bond_hidden, max_length):
    output = [Molecule(mol, cs, atom_hidden, bond_hidden, max_length) for idx, (mol, cs) in
              enumerate(tqdm(zip(mol_list, cs_list), total=len(mol_list)))]

    return MolDataSet(output)


def mol_collate_func(batch):
    adjacency_list, node_features_list, bond_features_list = [], [], []
    labels = []

    for molecule in batch:
        adjacency_list.append(molecule.adjacency_matrix)
        node_features_list.append(molecule.node_features)
        bond_features_list.append(molecule.bond_features)
        labels.append(molecule.label)

    return [torch.FloatTensor(features) for features in
            (adjacency_list, node_features_list, bond_features_list, labels)]


def construct_loader(mol_list, cs_list, batch_size, atom_hidden, bond_hidden, max_length, shuffle=True):
    dataset = construct_dataset(mol_list, cs_list, atom_hidden, bond_hidden, max_length)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=mol_collate_func,
                        shuffle=shuffle, drop_last=True, num_workers=0)
    return loader


if __name__ == '__main__':
    # load data
    element = '1H'
    with open('./Data/NMRShiftDB/preprocess/graph_' + element + '_train.pickle', 'rb') as f:
        [train_all_mol, train_all_cs] = pkl.load(f)
    with open('./Data/NMRShiftDB/preprocess/graph_' + element + '_test.pickle', 'rb') as f:
        [test_mol, test_cs] = pkl.load(f)

    max_length = max(max([data.GetNumAtoms() for data in train_all_mol]),
                     max([data.GetNumAtoms() for data in test_mol]))

    train_mol, valid_mol, train_cs, valid_cs = train_test_split(train_all_mol, train_all_cs, test_size=0.05,
                                                                random_state=np.random.randint(10000))

    atom_hidden = 115
    bond_hidden = 13
    train_loader = construct_loader(train_mol, train_cs, batch_size=32, atom_hidden=atom_hidden,
                                    bond_hidden=bond_hidden, max_length=max_length)
    test_loader = construct_loader(test_mol, test_cs, batch_size=32, atom_hidden=atom_hidden, bond_hidden=bond_hidden,
                                   max_length=max_length)

    for data in train_loader:
        [adjacency_matrix_list, node_features_list, bond_features_list, labels_list] = data
        batch_mask = torch.sum(torch.abs(node_features_list), dim=-1) != 0
        print(adjacency_matrix_list.shape)
        print(node_features_list.shape)
        print(bond_features_list.shape)
        print(batch_mask.int().shape)
        print(labels_list.shape)
        break
    print()
    for data in test_loader:
        [adjacency_matrix_list, node_features_list, bond_features_list, labels_list] = data
        print(adjacency_matrix_list.shape)
        print(node_features_list.shape)
        print(bond_features_list.shape)
        print(labels_list.shape)
        break
