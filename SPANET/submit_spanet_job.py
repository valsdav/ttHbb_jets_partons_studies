# i have a slurm cluster and I would like to submti jobs

import argparse
parser = argparse.ArgumentParser(description="Run the SPA-NET model")
parser.add_argument("-i", "--input", type=str, required=True, help="Input file with events")
parser.add_argument("-o", "--output", type=str, required=True, help="Output folder")
parser.add_argument("-m", "--model", type=str, required=True, help="Path to the SPA-NET model")
parser.add_argument("--jobs-folder", type=str, default="jobs", help="Folder for job scripts")
parser.add_argument("--nbatches-per-job", type=int, default=1000, help="N batches for job")
parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for parallel processing")
parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for the model")
parser.add_argument("--dry", action="store_true", help="Dry run, do not submit jobs")
args = parser.parse_args()

import os
import awkward as ak


def submit_job(job_name, input_file, output_folder, output_file, model_path, start_idx, stop_idx, num_workers, job_folder):
    job_script = f"""#!/bin/zsh
#SBATCH --job-name={job_name}
#SBATCH --output={output_folder}/{job_name}.out
#SBATCH --error={output_folder}/{job_name}.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={num_workers}
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=short


cd /work/dvalsecc/ttHbb/ttHbb_jets_partons_studies/SPANET
eval "$(micromamba shell hook --shell=zsh)"
micromamba activate pocket-coffea

python run_spanet.py \
    --input {input_file} \
    --output {output_file} \
    --model {model_path} \
    --start-index {start_idx} \
    --stop-index {stop_idx} \
    --batch-size {args.batch_size} \
    --num-workers {num_workers} 
echo 'Job {job_name} finished'
"""
    job_file = os.path.join(job_folder, f"{job_name}.sh")
    with open(job_file, "w") as f:
        f.write(job_script)
    # Submit the job
    if not args.dry:
        os.system(f"sbatch {job_file}")
    
    


# Load the input to count the number of events
data = ak.from_parquet(args.input)
nevets = len(data)
nevents_per_job = args.nbatches_per_job * args.batch_size
njobs = nevets // nevents_per_job + (1 if nevets % nevents_per_job > 0 else 0)

# Create the output folder if it doesn't exist
if not os.path.exists(args.output):
    os.makedirs(args.output)
if not os.path.exists(args.jobs_folder):
    os.makedirs(args.jobs_folder)
# Submit jobs
print(f"Creating {njobs} jobs")
for i in range(njobs):
    job_name = f"job_{i}"
    input_file = args.input
    output_folder = args.output
    output_file = os.path.join(args.output, f"output_{i}.parquet")
    model_path = args.model
    start_idx = i * nevents_per_job
    stop_idx = min((i + 1) * nevents_per_job, nevets)
    num_workers = args.num_workers
    submit_job(job_name, input_file, output_folder, output_file, model_path, start_idx, stop_idx, num_workers, args.jobs_folder)
