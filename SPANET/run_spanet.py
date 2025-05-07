import argparse
import awkward as ak
import numba
import numpy as np
import awkward as ak
import vector
vector.register_numba()
vector.register_awkward()
import os

# New assignment from Matej
from spanet_assignments import assign_prov_higgs_first
import spanet_predictions

parser = argparse.ArgumentParser(description="Run the SPA-NET model")
parser.add_argument("-i", "--input", type=str, required=True, help="Input file with events")
parser.add_argument("-o", "--output", type=str, required=True, help="Output file")
parser.add_argument("-m", "--model", type=str, required=True, help="Path to the SPA-NET model")
parser.add_argument("--start-index", type=int, default=-1, help="Start index for processing events")
parser.add_argument("--stop-index", type=int, default=-1, help="End index for processing events")
parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing events")
parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for parallel processing")
args = parser.parse_args()


############################################################################################################

@numba.njit
def assign_provenance_and_prob(t1pred, t2pred, hpred, 
                      t1prob, t2prob, hprob, njets):
    out = np.zeros((t1pred.shape[0], njets)) -1
    prob = np.zeros((t1pred.shape[0], njets, 3), dtype=np.float32)-np.float32(np.inf)
    
    #print(prob)
    for iev, (t1, t2, h, t1p, t2p, hp) in enumerate(zip(t1pred, t2pred, hpred, t1prob, t2prob, hprob)):
    
        if t1[0] == -2:
            prob_t1 = -np.inf
        else:
            prob_t1 = t1p[t1[0]][t1[1]][t1[2]]
            
        if t2[0] == -2:
            prob_t2 = -np.inf
        else:
            prob_t2 = t2p[t2[0]]
            
        if h[0] == -2:
            prob_h = -np.inf
        else:
            prob_h = hp[h[0]][h[1]]
            
        
        # prob_t1 = t1p[t1[0]][t1[1]][t1[2]]
        # prob_t2 = t2p[t2[0]][t2[1]]
        # prob_h = hp[h[0]][h[1]]
        # print(prob_t1, prob_t2, prob_h)
        
        for i in t1: 
            out[iev][i] = 1
            prob[iev][i][0] = prob_t1
        for i in t2: 
            out[iev][i] = 2
            prob[iev][i][1] = prob_t2
        for i in h: 
            out[iev][i] = 3
            prob[iev][i][2] = prob_h

    return out, prob


@numba.njit
def get_assignment_indices(prov, njets):
    h_idx = np.ones((prov.shape[0], 2), dtype=np.int8)*-1
    thad_w_idx = np.ones((prov.shape[0], 2), dtype=np.int8)*-1
    thad_b_idx = np.ones((prov.shape[0], 1), dtype=np.int8)*-1
    tlep_idx = np.ones((prov.shape[0], 1), dtype=np.int8)*-1
    
    for ev, jets in enumerate(prov):
        #indicies for higgs and thad filling 
        nh, nthad = 0,0
        for i, p in enumerate(jets):
            if p == 1:
                h_idx[ev, nh] = i
                nh +=1
            elif p == 2:
                thad_b_idx[ev,0] = i
            elif p== 3:
                tlep_idx[ev, 0] = i
            elif p== 5:
                thad_w_idx[ev, nthad] = i
                nthad+=1
    return  h_idx,thad_w_idx,thad_b_idx,tlep_idx

########################################################################################################

df = ak.from_parquet(args.input)

# Keeping only the indices requested
if args.start_index >= 0:
    if args.stop_index > len(df):
        args.stop_index = len(df)
    # restrict the dataset
    df = df[args.start_index:args.stop_index]

(jets,_,_,_,_,lep, met,_, weight) = ak.unzip(df)

import onnxruntime    # to inference ONNX models, we use the ONNX Runtime
sess_options = onnxruntime.SessionOptions()

sess_options.intra_op_num_threads = args.num_workers
sess_options.inter_op_num_threads = 1
#sess_options.intra_op_num_threads = 15
sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL


session = onnxruntime.InferenceSession(
    args.model,
    sess_options = sess_options,
    providers=['CPUExecutionProvider']
)


jets_padded = ak.fill_none(ak.pad_none(jets, 16, clip=True), {"btag":0., "pt":0., "phi":0., "eta":0.})

data = np.transpose(
    np.stack([
    ak.to_numpy(jets_padded.btag),
    ak.to_numpy(jets_padded.eta),
    ak.to_numpy(jets_padded.phi),
    #ak.to_numpy(jets_padded.pt),
    np.log(1 + ak.to_numpy(jets_padded.pt))
    
    ]),
    axes=[1,2,0]).astype(np.float32)

mask = ~ak.to_numpy(jets_padded.pt == 0)

met_data = np.stack([ak.to_numpy(met.eta),
                     ak.to_numpy(met.phi),
                     #ak.to_numpy(met.pt)
                     np.log(1+ ak.to_numpy(met.pt))
                    ], axis=1)[:,None,:].astype(np.float32)

lep_data = np.stack([ak.to_numpy(lep[:,0].eta),
                     ak.to_numpy(lep[:,0].phi),
                     #ak.to_numpy(lep[:,0].pt)
                     np.log(1+ ak.to_numpy(lep[:,0].pt))
                ], axis=1)[:,None,:].astype(np.float32)

ht_array = np.sum(ak.to_numpy(jets_padded.pt), axis=1)[:,None, None].astype(np.float32)

mask_global = np.ones(shape=[met_data.shape[0], 1]) == 1

njets_good = ak.sum(mask, axis=1)

batch_size = args.batch_size
nbatches = data.shape[0]// batch_size
print(f"{nbatches=}")

provenance = np.zeros((data.shape[0], 16))
provenance_no_overlap = np.zeros((data.shape[0], 16))

prob_assignment = np.zeros((data.shape[0], 16, 3), dtype=np.int8)
predictions_index = [ np.zeros((data.shape[0], 3), dtype=np.int8),
                     np.zeros((data.shape[0], 1), dtype=np.int8),
                     np.zeros((data.shape[0], 2), dtype=np.int8)
                    ]
predictions_index_nooverlap = [ 
                     np.zeros((data.shape[0], 2), dtype=np.int8), #thad W
                     np.zeros((data.shape[0], 1), dtype=np.int8), #thad b
                     np.zeros((data.shape[0], 1), dtype=np.int8), #tlep
                     np.zeros((data.shape[0], 2), dtype=np.int8)  #higgs
                    ]

for i in range(nbatches):
    start = i*batch_size
    if i < (nbatches-1):
        stop = start+batch_size
    else:
        stop = len(data)
    outputs = session.run(input_feed={
        "Source_data": data[start:stop],
        "Source_mask": mask[start:stop],
        "Met_data": met_data[start:stop],
        "Met_mask": mask_global[start:stop],
        "Lepton_data": lep_data[start:stop],
        "Lepton_mask": mask_global[start:stop],
        "ht_data": ht_array[start:stop], 
        "ht_mask": mask_global[start:stop]},
        output_names=["t1_assignment_log_probability", "t2_assignment_log_probability",
                     "h_assignment_log_probability"]
        )

    preds_nomask = spanet_predictions.extract_predictions(outputs[0:3], masking=False)
    prov, prob = assign_provenance_and_prob(*preds_nomask, outputs[0], outputs[1], outputs[2], 16)
    #print(prob)
    provenance[start:stop] = prov
    prob_assignment[start:stop]= prob
    for j in range(3):
       predictions_index[j][start:stop] = preds_nomask[j]
    
    prov_no_over, assign_order_counts = assign_prov_higgs_first(outputs)
    provenance_no_overlap[start:stop] = prov_no_over


    (h_idx_nooverlap, thad_w_idx_nooverlap,
     thad_b_idx_nooverlap, tlep_idx_nooverlap) = get_assignment_indices(prov_no_over, 16)
    predictions_index_nooverlap[0][start:stop] = thad_w_idx_nooverlap
    predictions_index_nooverlap[1][start:stop] = thad_b_idx_nooverlap
    predictions_index_nooverlap[2][start:stop] = tlep_idx_nooverlap
    predictions_index_nooverlap[3][start:stop] = h_idx_nooverlap
            

## Postprocessing
probabilities = np.exp(ak.to_numpy(prob_assignment))
# Factor two for symmetry reasons
probabilities[:,:,0] = probabilities[:,:,0] *2
probabilities[:,:,2] = probabilities[:,:,2] * 2

njets_good = ak.sum(mask, axis=1)
prob_assign_reshape = np.reshape(prob_assignment, (-1, 3) )
prob_masked = prob_assign_reshape[mask.flatten()]
prob_ak = ak.unflatten(ak.from_numpy(prob_masked), njets_good)

prov_overlap_ak = ak.unflatten(ak.from_numpy(provenance_no_overlap)[mask], njets_good)

df["prob_ak"] = prob_ak
df["jets"] = ak.zip( dict(zip(df.jets.fields,ak.unzip(df.jets))) |  {"prov_Thad": prob_ak[:,:,0],
                                                                     "prov_Tlep": prob_ak[:,:,1],
                                                                     "prov_H": prob_ak[:,:,2],
                                                                    })
df["jets"] = ak.with_field(df.jets, prov_overlap_ak, "prov_nooverlap")
df["higgs_predicted_indices"] = ak.from_numpy(predictions_index[2])
df["tophad_predicted_indices"] = ak.from_numpy(predictions_index[0])
df["toplep_predicted_indices"] = ak.from_numpy(predictions_index[1])

df["higgs_predicted_indices_nooverlap"] = ak.from_numpy(predictions_index_nooverlap[3])
df["tophad_w_predicted_indices_nooverlap"] = ak.from_numpy(predictions_index_nooverlap[0])
df["tophad_b_predicted_indices_nooverlap"] = ak.from_numpy(predictions_index_nooverlap[1])
df["toplep_predicted_indices_nooverlap"] = ak.from_numpy(predictions_index_nooverlap[2])

print(f"Saving output to: {args.output}")
ak.to_parquet(df, args.output)
