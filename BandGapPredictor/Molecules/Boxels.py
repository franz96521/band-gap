#!/usr/bin/env python
# coding: utf-8
"""
Simple boxel creation of molecules.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools

import numpy as np
import numpy.linalg as la
try:
    import pybel
except ImportError:
    pybel = None

import plotly.express as px

# Atom color rgb tuples (used for rendering, may be changed by users)
ATOM_COLORS = np.array([(0, 0, 0),  # Avoid atomic number to index conversion
                        (255, 255, 255), (217, 255, 255), (204, 128, 255),
                        (194, 255, 0), (255, 181, 181), (144, 144, 144),
                        (48, 80, 248), (255, 13, 13), (144, 224, 80),
                        (179, 227, 245), (171, 92, 242), (138, 255, 0),
                        (191, 166, 166), (240, 200, 160), (255, 128, 0),
                        (255, 255, 48), (31, 240, 31), (128, 209, 227),
                        (143, 64, 212), (61, 225, 0), (230, 230, 230),
                        (191, 194, 199), (166, 166, 171), (138, 153, 199),
                        (156, 122, 199), (224, 102, 51), (240, 144, 160),
                        (80, 208, 80), (200, 128, 51), (125, 128, 176),
                        (194, 143, 143), (102, 143, 143), (189, 128, 227),
                        (225, 161, 0), (166, 41, 41), (92, 184, 209),
                        (112, 46, 176), (0, 255, 0), (148, 255, 255),
                        (148, 224, 224), (115, 194, 201), (84, 181, 181),
                        (59, 158, 158), (36, 143, 143), (10, 125, 140),
                        (0, 105, 133), (192, 192, 192), (255, 217, 143),
                        (166, 117, 115), (102, 128, 128), (158, 99, 181),
                        (212, 122, 0), (148, 0, 148), (66, 158, 176),
                        (87, 23, 143), (0, 201, 0), (112, 212, 255),
                        (255, 255, 199), (217, 225, 199), (199, 225, 199),
                        (163, 225, 199), (143, 225, 199), (97, 225, 199),
                        (69, 225, 199), (48, 225, 199), (31, 225, 199),
                        (0, 225, 156), (0, 230, 117), (0, 212, 82),
                        (0, 191, 56), (0, 171, 36), (77, 194, 255),
                        (77, 166, 255), (33, 148, 214), (38, 125, 171),
                        (38, 102, 150), (23, 84, 135), (208, 208, 224),
                        (255, 209, 35), (184, 184, 208), (166, 84, 77),
                        (87, 89, 97), (158, 79, 181), (171, 92, 0),
                        (117, 79, 69), (66, 130, 150), (66, 0, 102),
                        (0, 125, 0), (112, 171, 250), (0, 186, 255),
                        (0, 161, 255), (0, 143, 255), (0, 128, 255),
                        (0, 107, 255), (84, 92, 242), (120, 92, 227),
                        (138, 79, 227), (161, 54, 212), (179, 31, 212),
                        (179, 31, 186), (179, 13, 166), (189, 13, 135),
                        (199, 0, 102), (204, 0, 89), (209, 0, 79),
                        (217, 0, 69), (224, 0, 56), (230, 0, 46),
                        (235, 0, 38), (255, 0, 255), (255, 0, 255),
                        (255, 0, 255), (255, 0, 255), (255, 0, 255),
                        (255, 0, 255), (255, 0, 255), (255, 0, 255),
                        (255, 0, 255)], dtype=np.float16)/255.0

# Atom numbers mapped to their symbols
ATOM_NUMBERS = {"H": 1, "HE": 2, "LI": 3, "BE": 4, "B": 5, "C": 6, "N": 7,
                "O": 8, "F": 9, "NE": 10, "NA": 11, "MG": 12, "AL": 13,
                "SI": 14, "P": 15, "S": 16, "CL": 17, "AR": 18, "K": 19,
                "CA": 20, "SC": 21, "TI": 22, "V": 23, "CR": 24, "MN": 25,
                "FE": 26, "CO": 27, "NI": 28, "CU": 29, "ZN": 30, "GA": 31,
                "GE": 32, "AS": 33, "SE": 34, "BR": 35, "KR": 36, "RB": 37,
                "SR": 38, "Y": 39, "ZR": 40, "NB": 41, "MO": 42, "TC": 43,
                "RU": 44, "RH": 45, "PD": 46, "AG": 47, "CD": 48, "IN": 49,
                "SN": 50, "SB": 51, "TE": 52, "I": 53, "XE": 54, "CS": 55,
                "BA": 56, "LA": 57, "CE": 58, "PR": 59, "ND": 60, "PM": 61,
                "SM": 62, "EU": 63, "GD": 64, "TB": 65, "DY": 66, "HO": 67,
                "ER": 68, "TM": 69, "YB": 70, "LU": 71, "HF": 72, "TA": 73,
                "W": 74, "RE": 75, "OS": 76, "IR": 77, "PT": 78, "AU": 79,
                "HG": 80, "TL": 81, "PB": 82, "BI": 83, "PO": 84, "AT": 85,
                "RN": 86, "FR": 87, "RA": 88, "AC": 89, "TH": 90, "PA": 91,
                "U": 92, "NP": 93, "PU": 94, "AM": 95, "CM": 96, "BK": 97,
                "CF": 98, "ES": 99, "FM": 100, "MD": 101, "NO": 102,
                "LR": 103, "RF": 104, "DB": 105, "SG": 106, "BH": 107,
                "HS": 108, "MT": 109, "DS": 110, "RG": 111, "CN": 112,
                "UUB": 112, "UUT": 113, "UUQ": 114, "UUP": 115, "UUH": 116,
                "UUS": 117, "UUO": 118}

# Atom valence radii in Å (used for bond calculation)
ATOM_VALENCE_RADII = np.array([0,  # Avoid atomic number to index conversion
                               230, 930, 680, 350, 830, 680, 680, 680, 640,
                               1120, 970, 1100, 1350, 1200, 750, 1020, 990,
                               1570, 1330, 990, 1440, 1470, 1330, 1350, 1350,
                               1340, 1330, 1500, 1520, 1450, 1220, 1170, 1210,
                               1220, 1210, 1910, 1470, 1120, 1780, 1560, 1480,
                               1470, 1350, 1400, 1450, 1500, 1590, 1690, 1630,
                               1460, 1460, 1470, 1400, 1980, 1670, 1340, 1870,
                               1830, 1820, 1810, 1800, 1800, 1990, 1790, 1760,
                               1750, 1740, 1730, 1720, 1940, 1720, 1570, 1430,
                               1370, 1350, 1370, 1320, 1500, 1500, 1700, 1550,
                               1540, 1540, 1680, 1700, 2400, 2000, 1900, 1880,
                               1790, 1610, 1580, 1550, 1530, 1510, 1500, 1500,
                               1500, 1500, 1500, 1500, 1500, 1500, 1600, 1600,
                               1600, 1600, 1600, 1600, 1600, 1600, 1600, 1600,
                               1600, 1600, 1600, 1600, 1600, 1600],
                              dtype=np.float32)/1000.0
# Prevent unintentional changes
ATOM_VALENCE_RADII.flags.writeable = False

# Atom radii in Å (used for rendering, scaled down by factor 0.4, may be
# changed by users)
ATOM_RADII = np.array(ATOM_VALENCE_RADII, copy=True)*0.4

# Bond radius in Å (used for rendering, may be changed by users)
BOND_RADIUS = 0.1




class Molecule(object):

    def __init__(self, atomic_numbers, positions):
        self.atomic_numbers = atomic_numbers
        self.positions = positions
        self.bonds = None
        self._atomic_radii = None

    @property
    def atomic_radii(self):
        if self._atomic_radii is None:
            self._atomic_radii = ATOM_RADII[self.atomic_numbers]
        return self._atomic_radii

    def calculate_bonds(self, method='radii', param=None):
        """
        This function calculates which pair of atoms forms an atomic bond.
        Pairs of indices are returned, indicating that the two associated atoms
        form a bond.

        The method used for this is specified with the parameter 'method'.
        Currently two methods are supported:
         - using the valence radii of the atoms (optionally multiplied by
             param)
         - using a constant delta as maximum distance between to bonded atoms
        """
        method = method.lower()
        if method not in ('radii', 'constant_delta'):
            raise ValueError("method should be either 'radii' or 'constant "
                             "delta', not '%s'" % method)

        if self.bonds is not None:
            # Bonds were calculated for this molecule already.
            # Reuse those, if the method and optional parameter are like those
            # provided to this function.
            if self.bonds.method == method:
                if self.bonds.param == param:
                    return
                elif param is None:
                    # Allow reuse when default param is used.
                    if method == 'radii' and self.bonds.param == 1.0:
                        return

        if method == 'radii':
            radii = ATOM_VALENCE_RADII[self.atomic_numbers]
            if param is None:
                param = 1.0

        if method == 'constant_delta':
            if param is None:
                raise ValueError("method 'constant_delta' requires param to "
                                 "specify the maximum distance between two "
                                 "atoms to cause a bond between them.")

        index_pair_list = []
        for index, position in enumerate(self.positions):
            distance_vectors = self.positions[index+1:] - position
            distances_squared = np.sum(np.power(distance_vectors, 2), axis=1)
            if method == 'radii':
                deltas = np.square((radii[index+1:]+radii[index])*param)
            elif method == 'constant_delta':
                deltas = param**2
            nonzero_indices = np.nonzero(distances_squared <= deltas)
            num_bonds = len(nonzero_indices[0])
            if num_bonds > 0:
                index_pairs = np.hstack((np.ones(num_bonds, dtype=np.uint32)
                                         .reshape(num_bonds, 1)*index,
                                         index+1+nonzero_indices[0]
                                         .reshape(num_bonds, 1)))
                index_pair_list.append(index_pairs)
        if index_pair_list:
            index_pairs = np.vstack(index_pair_list)
        else:
            index_pairs = np.zeros((0, 2))
        self.bonds = Bonds(method, param, index_pairs)


class Bonds(object):
    """
    A simple class for storing information about atomic bonds in a molecule.
    """

    def __init__(self, method, param, index_pairs):
        self.method = method
        self.param = param
        self.index_pairs = index_pairs

    def __len__(self):
        return len(self.index_pairs)


class UnknownFileFormatException(Exception):
    """
    This exception will be raised when read() is called with file name that
    cannot be mapped to a known file format.
    """
    pass


def read(file_name, file_format=None):
    """
    Reads molecules from a file. If available, pybel will be used to read the
    file, otherwise only xyz files can be read.
    If the file_format is unknown, a UnknownFileFormatException will be raised.
    """
    if pybel:
        if file_format is None:
            file_extension = file_name.split('.')[-1]
            if file_extension in pybel.informats.keys():
                file_format = file_extension
            else:
                message = ("The file format {format} is  not supported by "
                           "pybel. If the file extension does not match the "
                           "actual format, please use the file_format "
                           "parameter.".format(format=file_extension))
                raise UnknownFileFormatException(message)
        elif file_format not in pybel.informats.keys():
            message = ("The file format {format} is not supported by pybel, so"
                       "mogli cannot understand it yet. Sorry!"
                       .format(format=file_format))
            raise UnknownFileFormatException(message)
        try:
            # Try reading the file with the 'b' option set. This speeds up
            # file reading dramatically for some formats (e.g. pdb).
            # If this fails for some reasons, use pybel.readfile().
            molecules = []
            conv = pybel.ob.OBConversion()
            conv.SetOptions("b".encode('ascii'), conv.INOPTIONS)
            success = conv.SetInFormat(file_format.encode('ascii'))
            if success:
                mol = pybel.ob.OBMol()
                has_molecule = conv.ReadFile(mol, file_name.encode('ascii'))
                while has_molecule:
                    num_atoms = mol.NumAtoms()
                    atomic_numbers = np.zeros(num_atoms, dtype=np.uint8)
                    positions = np.zeros((num_atoms, 3), dtype=np.float32)
                    for i, atom in enumerate(pybel.ob.OBMolAtomIter(mol)):
                        atomic_numbers[i] = atom.GetAtomicNum()
                        positions[i] = (atom.GetX(), atom.GetY(), atom.GetZ())
                    molecules.append(Molecule(atomic_numbers, positions))
                    has_molecule = conv.Read(mol)
        except:
            molecule_file = pybel.readfile(file_format.encode('ascii'),
                                           file_name.encode('ascii'))
            molecules = []
            for molecule in molecule_file:
                atoms = molecule.atoms
                atomic_numbers_iterator = (atom.atomicnum for atom in atoms)
                atomic_numbers = np.fromiter(atomic_numbers_iterator,
                                             dtype=np.uint8, count=len(atoms))
                positions_iterator = itertools.chain.from_iterable(atom.coords
                                                                   for atom
                                                                   in atoms)
                positions = np.fromiter(positions_iterator,
                                        dtype=np.float32, count=len(atoms)*3)
                positions.shape = (len(atoms), 3)
                molecules.append(Molecule(atomic_numbers, positions))
        return molecules
    else:
        if (file_format == 'xyz' or
                (file_format is None and file_name.endswith('.xyz'))):
            with open(file_name, 'r') as input_file:
                file_content = input_file.readlines()
            molecules = []
            while file_content:
                num_atoms = int(file_content[0])
                atom_strings = file_content[2:2+num_atoms]
                file_content = file_content[2+num_atoms:]
                atomic_numbers = [ATOM_NUMBERS[atom_string.split()[0].upper()]
                                  for atom_string
                                  in atom_strings]
                positions = np.array([(float(atom_string.split()[1]),
                                       float(atom_string.split()[2]),
                                       float(atom_string.split()[3]))
                                      for atom_string
                                      in atom_strings])
                molecules.append(Molecule(atomic_numbers, positions))
            return molecules
        else:
            message = ("Failed to import pybel. Currently mogli only supports "
                       "xyz files if pybel is not installed.")
            raise UnknownFileFormatException(message)




def create(molecule, size = 128,scale = 15,show_bonds=True, bonds_method='radii', bonds_param=1.5):
    if show_bonds:
        molecule.calculate_bonds(bonds_method, bonds_param)

    molecule.positions = molecule.positions - np.min(molecule.positions, axis=0)
    molecule.positions = molecule.positions / np.max(molecule.positions, axis=0)

    molecule.positions = molecule.positions * size
    # print (molecule.positions)
    world = np.zeros((size,size,size,3))
    world.fill(-1)
    index_pairs = molecule.bonds.index_pairs
    for bond in index_pairs:
        p0 = molecule.positions[bond[0]]
        p1 = molecule.positions[bond[1]]
        R = .5
        v = p1 - p0
        mag = np.linalg.norm(v)
        v = v / mag
        not_v = np.array([1, 0, 0])
        if (v == not_v).all():
            not_v = np.array([0, 1, 0])
        n1 = np.cross(v, not_v)
        n1 /= np.linalg.norm(n1)
        n2 = np.cross(v, n1)
        t = np.linspace(0, mag, 100)
        theta = np.linspace(0, 2 * np.pi, 100)
        t, theta = np.meshgrid(t, theta)
        X, Y, Z = [p0[i] + v[i] * t + R * np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]
        # remove rows with nan
        points = np.array([X.flatten(),Y.flatten(),Z.flatten()]).T
        points = points[~np.isnan(points).any(axis=1)]

        for x,y,z in points:            
            x,y,z = int(x),int(y),int(z)            
            if x >= 0 and x < size and y >= 0 and y < size and z >= 0 and z < size:
                world[x,y,z] = np.array([0,0,0])


    for i , atom in enumerate(molecule.positions) :
        x,y,z = atom.astype(int)
        color = ATOM_COLORS[molecule.atomic_numbers[i]]
        radius = (ATOM_RADII[molecule.atomic_numbers[i]]*scale).astype(int)
        # plot sphere around atom 
        for x1 in range(x-radius,x+radius):
            for y1 in range(y-radius,y+radius):
                for z1 in range(z-radius,z+radius):
                    if (x1-x)**2 + (y1-y)**2 + (z1-z)**2 < radius**2:
                        # check if inside world
                        if x1 >= 0 and x1 < size and y1 >= 0 and y1 < size and z1 >= 0 and z1 < size:
                            world[x1,y1,z1] = color

    return world


def showBoxels(molecule):
    x=[]
    y=[]
    z=[]
    color =[]
    for i in range(molecule.shape[0]):
        for j in range(molecule.shape[1]):
            for k in range(molecule.shape[2]):
                if molecule[i,j,k].sum() != -1*3:
                    x.append(i)
                    y.append(j)
                    z.append(k)
                    color.append(molecule[i,j,k].sum())
    # print(set(color))
    fig = px.scatter_3d(x=x, y=y, z=z, color=color, height=1000)

    fig.show()