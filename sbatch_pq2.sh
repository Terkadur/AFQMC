#!/bin/sh
#SBATCH -t 6:00:00
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem=2G
#SBATCH --account=default
#SBATCH -e ./output/pq2_%j.err
#SBATCH -o ./output/pq2_%j.out

ulimit -s unlimited

module load julia

time julia --project=@. job_pq2.jl $1 $2

