#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --time 07:00:00
#SBATCH --mem=55gb
#SBATCH --ntasks-per-node 10
#SBATCH --mail-type=begin        # send mail when process begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=antonio.acquaviva@studenti.unipr.it

#< Charge resources to account
#SBATCH --account G_VBD

# /hpc/scratch/antonio.acquaviva/work_dir/ub_output


module load miniconda3
module load cuda/11 gnu8.2.0/8.2.0
source '/hpc/share/tools/miniconda3/etc/profile.d/conda.sh'

conda activate detectron2

CFG_NAME="instagram50.0"

cd /hpc/home/antonio.acquaviva/unbiased-teacher

python train_net.py --num-gpus 4 --config configs/coco_supervision/faster_rcnn_R_50_FPN_sup10_run1-${CFG_NAME}.yaml OUTPUT_DIR output/${CFG_NAME}