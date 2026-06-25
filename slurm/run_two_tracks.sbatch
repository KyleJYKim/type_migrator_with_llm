cat > slurm/train_two_tracks.sbatch << 'SBATCH'
#!/bin/bash
#SBATCH --job-name=train-both-tracks
#SBATCH --qos=normal
#SBATCH --gpus=a100_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

CONFIG="${1:-configs/qwen7b_qlora.yaml}"
TRACKS=(track1_no_gradual track2_both_pass)

eval "$(conda shell.bash hook)"
conda activate etr

cd ~/elixir_types_trainer/type_migrator_with_llm

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for t in "${TRACKS[@]}"; do
  echo "==================  TRAIN  ${t}  =================="
  python scripts/train_sft.py \
    --config "$CONFIG" \
    --data_dir "data/${t}" \
    --output_dir "runs/${t}"
done

SBATCH