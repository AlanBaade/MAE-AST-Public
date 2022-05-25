# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class MAE_AST_Criterion_Config(FairseqDataclass):
    reconstruction_weight: float = field(
        default=10.0,
        metadata={"help": "weight for reconstruction in SSAST Type Model. Equals Lambda. Default 10. Set to 0 to not use"},
    )
    classification_weight: float = field(
        default=1.0,
        metadata={"help": "weight for classification in SSAST Type Model. Default 1. Set to 0 to not use"},
    )


@register_criterion("mae_ast", dataclass=MAE_AST_Criterion_Config)
class MAE_AST_Criterion(FairseqCriterion):
    def __init__(
            self,
            task,
            reconstruction_weight,
            classification_weight,
            # log_keys=None,
    ):
        super().__init__(task)
        self.reconstruction_weight = reconstruction_weight
        self.classification_weight = classification_weight

    def forward(self, model, sample, reduce=True, log_pred=False):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        net_output = model(**sample["net_input"])

        loss = 0.0
        logging_output = {}

        logp_m_list_recon, logp_m_list_class = model.get_logits(net_output)
        targ_m_list = model.get_targets(net_output, True)
        assert (self.reconstruction_weight > 0 or self.classification_weight > 0) and len(logp_m_list_recon) > 0

        if self.reconstruction_weight > 0:
            loss_recon = F.mse_loss(logp_m_list_recon, targ_m_list)
            logging_output["loss_recon"] = loss_recon.detach().item()
            loss += self.reconstruction_weight * loss_recon

        if self.classification_weight > 0:
            all_dots = torch.matmul(logp_m_list_class, targ_m_list.transpose(-1, -2))
            log_softmax = torch.log_softmax(all_dots, dim=-1)
            loss_info_nce = -torch.mean(torch.diagonal(log_softmax, dim1=-2, dim2=-1))

            logging_output["loss_info_nce"] = loss_info_nce.detach().item()

            loss += self.classification_weight * loss_info_nce

        sample_size = 1

        logging_output = {
            "loss": loss.item() if reduce else loss,
            "ntokens": targ_m_list.size(1),
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
            **logging_output,
        }

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training (copied from normal cross entropy)."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

        counts = {}
        for lk in logging_outputs[0].keys():
            if lk.startswith("count_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val)
                counts[lk] = val

        for lk in logging_outputs[0].keys():
            if lk.startswith("loss_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val / sample_size / math.log(2), round=3)
            elif lk.startswith("correct_"):
                val = sum(log[lk] for log in logging_outputs)
                metrics.log_scalar(lk, val / counts[re.sub("correct", "count", lk)])

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        raise NotImplementedError()

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
