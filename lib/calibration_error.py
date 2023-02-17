"""
References: 
    https://github.com/baal-org/baal/blob/master/baal/utils/metrics.py
"""

import math
from typing import Any

import numpy as np
from ignite.metrics import Metric
from lib.utils import to_prob
import matplotlib.pyplot as plt


class ECE(Metric):
    """
    Expected Calibration Error (ECE)
    References:
        https://arxiv.org/pdf/1706.04599.pdf
    """

    def __init__(self, n_bins=10, **kwargs):
        self.n_bins = n_bins  # number of bins to discretize the uncertainty.
        self.tp = None  # true positive
        self.samples = None  # num samples per bin
        self.conf_agg = None  # accumulate confidence
        super(ECE, self).__init__()

    def update(self, output) -> None:
        y_pred = output[0].detach().cpu().numpy()  # logits or predictions of model
        y = output[1].detach().cpu().numpy()  # labels
        y_pred = to_prob(y_pred)

        # this is to make sure handling 1.0 value confidence to be assigned to a bin
        y_pred = np.clip(y_pred, 0, 0.9999)

        for pred, t in zip(y_pred, y):
            conf = pred.max()
            p_cls = pred.argmax()

            bin_id = int(math.floor(conf * self.n_bins))
            self.samples[bin_id] += 1
            self.tp[bin_id] += int(p_cls == t)
            self.conf_agg[bin_id] += conf

    def _acc(self):
        return self.tp / np.maximum(1, self.samples)

    def calculate_result(self):
        n = self.samples.sum()
        average_confidence = self.conf_agg / np.maximum(self.samples, 1)
        return ((self.samples / n) * np.abs(self._acc() - average_confidence)).sum()

    def reset(self) -> None:
        self.tp = np.zeros([self.n_bins], dtype=int)  # true positive
        self.samples = np.zeros([self.n_bins], dtype=int)
        self.conf_agg = np.zeros([self.n_bins])

    def compute(self) -> Any:
        return self.calculate_result()

    def plot(self, pth=None):
        """
        Plot each bins, ideally this would be a diagonal line.
        Args:
            pth: if provided the figure will be saved under the given path
        """
        # Plot the ECE
        plt.bar(np.linspace(0, 1, self.n_bins), self._acc(), align="edge", width=0.1)
        plt.plot([0, 1], [0, 1], "--", color="tab:gray")
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.ylabel("Accuracy")
        plt.xlabel("Uncertainty")
        plt.grid()

        if pth:
            plt.savefig(pth)
            plt.close()
        else:
            plt.show()
