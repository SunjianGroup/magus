# Example Lists
|   number   |    target    |
|  :-------:    |    :-------:   |
| 01 | [Generate structures](#generate-structures)  |
| 02 | [Relax structures](#relax-structures)  |
| 03 | [3D bulk search](#3d-bulk-search) |
| 04 | [Molecule crystal search](#molecule-crystal-search)    |
| 05 | [Cluster search](#cluster-search)          |
| 06 | [Surface reconstruct](#surface-reconstruct) |
| 07 | [Machine learning search](#machine-learning-search)    | 
| 08 | [2D bulk search](#2d-bulk-search) | 

## Generate structures
- 01--1-B12: Generate 3d periodic crystal structures of boron with 12 atoms per unit cell by symmetry.

- 01--2-NH4NO3: Generate 3d periodic crystal structures of molecule crystal with 8 NH4 and NO3 molecules per unit cell by symmetry Pccn (56).

## Relax structures
- 02--1-C8-VASP: Structure relaxation of diamond by vasp interface. 
- 02--2-B12-MTP: Relax 2000 structures with MTP and VASP.

## 3D bulk search
- 03--1-Al-fix-EMT: GAsearch of fixed composition Al (12 atoms per cell) by EMT.
- 03--2-Al-fix-VASP: GAsearch of fixed composition Al (12 atoms per cell) by VASP.
- 03--3-TiO2-fix-VASP: GAsearch of fixed composition TiO2 (12 atoms per cell).
- 03--4-MgAlO-fix-GULP: GAsearch of fixed composition MgAlO under high pressure by GULP.
- 03--5-Si-fix-Castep: GAsearch of fixed composition Si by Castep.
- 03--6-ZnOH-var-GULP: GAsearch of variable composition Znx(OH)y.

## Molecule crystal search
- 04--1-CH4-fix-VASP: GAsearch of molecule crystal CH4 with 4 molecules per unit cell.

## Cluster search
- 05--1-LJ26: Ground state of Lennard-Jones cluster of 26 atoms.
## Surface reconstruct
- 06--1-C_2x1_100: Surface reconstruction of diamond (100)-2×1.
- 06--2-SnO2_4x1_110: Surface reconstruction of SnO2 (110)-4×1.
## Machine learning search
- 07--1-MgSiO3-MTP: Use mtp to search MgSiO3 under high pressure.
- 07--2-Na-NEP: A toy example for use NEP to search sodium.
## 2D-bulk-search
- 08--1-graphene: GAsearch of 2D carbon (graphene) by VASP.