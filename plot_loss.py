'''
keras callback to plot loss
'''
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
from typing import *
from math import ceil
import os


class LossPlotter():
    def __init__(self, metrics, batch_mode=False, output_dir=os.getcwd()):
        self.metrics = {m:[] for m in metrics}
        self.nmetrics = len(self.metrics)
        self.batch_mode = batch_mode
        self.output_dir = output_dir
        self.i = 0
        
    def on_epoch_end(self, metrics:Dict, plot=True):
        for m,v in metrics.items():
            self.metrics[m].append(v)
        self.i += 1
        if plot:
            self.performance_plot()
        # self.save_metrics()

    # def performance_save(self, logs):
    #     self.logs.append(logs)
        
    #     for l in self.loss_labels:
    #         self.losses[l].append(logs.get(l))
    #         self.val_losses[l].append(logs.get("val_" + l))
    #     self.i += 1

       
    def performance_plot(self):
        nrows = ceil(self.nmetrics/4)
        self.figure, axs = plt.subplots(nrows, 4, figsize=(24,nrows*5), dpi=100)
        # self.figure.tight_layout()
        for il, l in enumerate(self.metrics.keys()):
            if nrows>1:
                ax = axs[il // 4][il %4]
            else:
                ax = axs[il %4]
            ax.set_title(l)
            ax.plot(np.arange(len(self.metrics[l])), self.metrics[l], ".-", label=l)
            if "loss" in l:
                ax.set_yscale("log")
            ax.set_xlabel("epochs")
            ax.legend()
        
        if not self.batch_mode:
            clear_output(wait=True)
            plt.show()
        self.figure.savefig(self.output_dir+ "/loss_plot.png")
        plt.close(self.figure)


    # def save_metrics(self):
    #     metrics = {}
    #     for l,v in self.losses.items():
    #         metrics[l] = v
    #     for l,v in self.val_losses.items():
    #         metrics[f"val_{l}"] = v
    #     df = pd.DataFrame(metrics)
    #     df.to_csv(self.output_dir + "/metrics_history.csv")
        

    # def save_figure(self, fname):
    #     if self.batch_mode:
    #         self.performance_plot()
    #     self.figure.savefig(self.output_dir + '/' + fname)
    #     plt.close(self.figure)
