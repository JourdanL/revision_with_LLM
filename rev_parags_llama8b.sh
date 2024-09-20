#! /bin/bash
#SBATCH --job-name=test_instruct_gen
#SBATCH --output=test_instruct_gen%j.out
#SBATCH --error=test_instruct_gen%j.err
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=10
#SBATCH --partition=gpu_p13
#SBATCH --gres=gpu:2
#SBATCH --time=11:00:00
#SBATCH --account=joh@v100
#SBATCH -C v100-32g

module purge # nettoyer les modules herites par defaut
conda deactivate # desactiver les environnements herites par defaut
module load pytorch-gpu/py3/2.2.0 # charger les modules
set -x # activer l’echo des commandes
srun python -u rev_parags_llama8b.py
