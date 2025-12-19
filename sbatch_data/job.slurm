#!/bin/bash
#SBATCH -J test
#SBATCH -p first
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=200G
#SBATCH -o d_z0.o%j
#SBATCH -e d_z0.e%j
##SBATCH --nodelist=tms03s
##SBATCH --exclusive


cd /stu02/shiqc24/d_z0_ML/quanliucheng/
python3 main.py