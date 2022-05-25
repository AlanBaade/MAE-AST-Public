# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from dataclasses import dataclass, field
from fairseq.data import Dictionary
from mae_ast.data import MAE_AST_Dataset
from fairseq.dataclass import ChoiceEnum
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from omegaconf import MISSING

logger = logging.getLogger(__name__)

MASK_TYPE_CHOICES = ChoiceEnum(["retain_spans", "random_mask", "random_mask_batched", "chunk_mask"])


@dataclass
class MAE_AST_Pretraining_Config(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})

    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down "
                    "sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=False,
        metadata={"help": "pad shorter samples instead of cropping"},
    )
    max_keep_size: Optional[int] = field(
        default=None,
        metadata={"help": "exclude sample longer than this"},
    )
    max_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "max sample size to crop to for batching"},
    )
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "min sample size to crop to for batching"},
    )
    random_crop: Optional[bool] = field(
        default=True,
        metadata={"help": "always crop from the beginning if false"},
    )
    pad_audio: Optional[bool] = field(
        default=False,
        metadata={"help": "pad audio to the longest one in the batch if true"},
    )

    feature_type: Optional[str] = field(
        default='wav',
        metadata={"help": "choose from ['wav', 'spectrogram', 'fbank', 'mfcc']"}
    )

    feature_rate: Optional[int] = field(
        default=100,
        metadata={
            "help": "rate of feature input to the transformer, if use wav, this arg is omited, else if use spectrogram/fbank/mfcc, the default is 100, i.e. 1s audio gives 100 frames. the label rate of using MFCC is also 100"}
    )

    feature_dim: Optional[int] = field(
        default=100,
        metadata={
            "help": "dim feature input to the transformer, if use wav, this arg is omited, else if use spectrogram/fbank/mfcc, the default is 80"}
    )

    deltas: Optional[bool] = field(
        default=True,
        metadata={
            "help": "whether or not add delta and delta-delta to the feature, only effective for spectrogram/fbank/mfcc"}
    )

    mask_spans: Optional[bool] = field(
        default=False,
        metadata={"help": "mask random spans, same as that is used in HuBERT and w2v2"}
    )

    mask_type: MASK_TYPE_CHOICES = field(
        default='random_mask',
        metadata={"help":
                      """Determine type of mask for MAE pretraining. 
                      -retain_spans: Only for frame data. Wav2Vec2 like masking.
                      -random_mask: Perform masking on completely random tokens. No chunking. Used in MAE
                      -random_mask_batched: random_mask with the same mask across the batch.
                      -chunk_mask: Perform masking on chunks until mask_spans hit. From SSAST. Same across batch for speed.
                          """}
    )


@register_task("mae_ast_pretraining", dataclass=MAE_AST_Pretraining_Config)
class MAE_AST_Pretraining_Task(FairseqTask):
    cfg: MAE_AST_Pretraining_Config

    def __init__(
            self,
            cfg: MAE_AST_Pretraining_Config,
    ) -> None:
        super().__init__(cfg)

        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"MAEPretrainingTask Config {cfg}")

        self.cfg = cfg

    @property
    def source_dictionary(self) -> Optional[Dictionary]:
        return None

    @property
    def target_dictionary(self) -> Optional[Dictionary]:
        return None

    @property
    def dictionaries(self) -> List[Dictionary]:
        return None

    @classmethod
    def setup_task(
            cls, cfg: MAE_AST_Pretraining_Config, **kwargs
    ) -> "MAE_AST_Pretraining_Task":
        return cls(cfg)

    def load_dataset(self, split: str, **kwargs) -> None:
        manifest = f"{self.cfg.data}/{split}.tsv"

        self.datasets[split] = MAE_AST_Dataset(
            manifest,
            sample_rate=self.cfg.sample_rate,
            max_keep_sample_size=self.cfg.max_keep_size,
            min_keep_sample_size=self.cfg.min_sample_size,
            max_sample_size=self.cfg.max_sample_size,
            pad_audio=self.cfg.pad_audio,
            normalize=self.cfg.normalize,
            random_crop=self.cfg.random_crop,
            feature_type=self.cfg.feature_type,
            feature_dim=self.cfg.feature_dim,
            deltas=self.cfg.deltas,
            feature_rate=self.cfg.feature_rate
        )

    def max_positions(self) -> Tuple[int, int]:
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(self, indices: np.array, *args, **kwargs) -> np.array:
        return indices
