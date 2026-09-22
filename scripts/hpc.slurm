#!/usr/bin/env bash
#SBATCH --job-name=taxcalcbench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=taxcalcbench-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [ "$#" -eq 0 ]; then
    echo "Usage: sbatch scripts/hpc.slurm configs/<country>.json [...] [--single-family]" >&2
    exit 1
fi

# Activate the installed Python environment before sbatch. No GPU is needed.
# Keep roots on persistent shared storage, not a node's temporary filesystem.
export PYTHONUNBUFFERED=1
srun "${TAXCALCBENCH_PYTHON:-python}" -m taxcalcbench run \
    --source-root "${TAXCALCBENCH_DATA_ROOT:-data}" \
    --output-root "${TAXCALCBENCH_OUTPUT_ROOT:-outputs/full}" \
    --countries "$@"
