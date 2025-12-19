#!/bin/bash
#SBATCH -N 1

date

#yhrun -n 1 -p work ./phimh_scale.x
./RealUrbanR.x

date
