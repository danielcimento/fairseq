# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
import json

from typing import Optional
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import register_criterion

from fairseq.criterions.cross_entropy import (
    CrossEntropyCriterion,
    CrossEntropyCriterionConfig,
)


@dataclass
class LMDistillationCriterionConfig(CrossEntropyCriterionConfig):
    kd_args: Optional[str] = field(
        default=None,
        metadata={"help": "arguments for knowledge distillation (kd_strategy)"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )


@register_criterion(
    "lm_distillation_loss",
    dataclass=LMDistillationCriterionConfig,
)
class LMDistillationCriterion(CrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        kd_args=None,
        ignore_prefix_size=0,
        report_accuracy=False,
        length_normalize_loss=False,
    ):
        super().__init__(task, sentence_avg)
        self.sentence_avg = sentence_avg
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.length_normalize_loss = length_normalize_loss

        # new parameters
        assert (
            kd_args is not None
        ), "Knowledge distillation arguments (lambda) are missing!"

        kd_args = json.loads(kd_args)

        self.lambd = kd_args.get(
            "lambda", 1.0
        )  # lambda for ratio of KD loss and NLL loss

        assert self.lambd > 0.0 and self.lambd <= 1.0, "lambda should be in (0, 1]."

    def forward(self, model, teacher_model, sample, update_num=None, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])

        assert (
            teacher_model is not None
        ), "knowledge distillation requires a teacher model!"

        # compute teacher outputs
        # make sure to wrap it in a torch.no_grad()
        # as we want the teacher model only on eval mode
        # and not generate any gradients for itself
        with torch.inference_mode():
            teacher_output = teacher_model(**sample["net_input"])
            sample["teacher_output"] = teacher_output

        loss, extra = self.compute_loss(
            model,
            net_output,
            sample,
            teacher_model=teacher_model,
            teacher_output=teacher_output,
        )

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "kd_loss": (
                extra["kd_loss"].data if extra.get("kd_loss", None) is not None else 0
            ),
        }

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample, log_probs=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=log_probs)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs, target

    # helper function to compute the kd loss
    def compute_kd_loss(self, model, net_output, sample, teacher_model, teacher_output):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        teacher_lprobs, _ = self.get_lprobs_and_target(
            teacher_model, teacher_output, sample
        )
        pad_mask = target.eq(self.padding_idx).unsqueeze(-1)

        kd_loss = (
            F.kl_div(lprobs, teacher_lprobs, log_target=True, reduction="none")
            .masked_fill_(pad_mask, 0.0)
            .sum(dim=-1)
        )

        token_lens = (
            target.ne(self.padding_idx).sum(dim=1, keepdim=True).clamp(min=1).float()
        )

        kd_loss = (
            (kd_loss.sum(dim=1) / token_lens)
            if self.length_normalize_loss
            else kd_loss.sum(dim=1)
        ).sum()

        return kd_loss, lprobs, target, token_lens

    def compute_loss(
        self, model, net_output, sample, teacher_model=None, teacher_output=None
    ):
        kd_loss, lprobs, target, token_lens = self.compute_kd_loss(
            model, net_output, sample, teacher_model, teacher_output
        )

        # compute preliminary nll_loss of student_model
        nll_loss = F.nll_loss(
            input=lprobs.transpose(-2, -1),
            target=target,
            reduction="none",
            ignore_index=self.padding_idx,
        )

        nll_loss = (
            (nll_loss.sum(dim=1) / token_lens)
            if self.length_normalize_loss
            else nll_loss.sum(dim=1)
        ).sum()

        loss = self.lambd * kd_loss + (1 - self.lambd) * nll_loss
        return loss, {"kd_loss": kd_loss, "nll_loss": nll_loss}

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        # sum metrics
        loss = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        kd_loss = sum(log.get("kd_loss", 0) for log in logging_outputs)
        # log metrics
        metrics.log_scalar(
            "loss", loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar("kd_loss", kd_loss / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: (
                    round(meters["n_correct"].sum * 100.0 / meters["total"].sum, 3)
                    if meters["total"].sum > 0
                    else float("nan")
                ),
            )
