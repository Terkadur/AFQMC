#!/bin/sh
#SBATCH -t 1:00:00
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem=4G
#SBATCH --account=default
#SBATCH -e ./output/etgent_%j.err
#SBATCH -o ./output/etgent_%j.out

ulimit -s unlimited

module load julia

time julia --project=@. job_etgent.jl $2 $1
