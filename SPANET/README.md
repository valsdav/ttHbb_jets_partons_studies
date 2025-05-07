# How to run SPANeT

python submit_spanet_job.py -i /work/dvalsecc/ttHbb/ttHbb_jets_partons_studies/sig_bkg_30_08_2023_v2/out_forTraining_sig_bkg -o `pwd`/output -m spanet-gpu3.onnx --nevents-per-job 20000 --num-workers 4 --batch-size 512 --dry
