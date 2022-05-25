# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#	Adopted from Fast VGS Wrapper by Puyuan Peng
"""*********************************************************************************************"""
#   FileName     [ upstream/mae_ast/expert.py ]
#   Synopsis     [ Upstream MAE-AST Wrapper ]
#   Author       [ Alan Baade ]
#   Copyright    [ Copyleft(c), Alan Baade ]
"""*********************************************************************************************"""

import argparse
from typing import List
from packaging import version

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import torchaudio

from ..interfaces import UpstreamBase

import sys
sys.path.insert(1, '/home/abaade/MAE-AST-Public')
from mae_ast.models.mae_ast import MAE_AST, MAE_AST_Config
from mae_ast.tasks.mae_ast_pretraining import MAE_AST_Pretraining_Config, MAE_AST_Pretraining_Task

from types import SimpleNamespace

class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)
        
        checkpoint = torch.load(ckpt)
        
        self.cfg = checkpoint["cfg"]["model"]
        self.task_cfg = checkpoint["cfg"]["task"]
        
        self.model = MAE_AST(SimpleNamespace(**checkpoint["cfg"]["model"]), SimpleNamespace(**checkpoint["cfg"]["task"]))
        
        self.model.load_state_dict(checkpoint["model"], strict=True)

        # Required for hidden states to have defined indices.
        self.model.encoder.layerdrop = 0

        self.sample_rate = self.task_cfg['sample_rate']
        self.feature_dim = self.task_cfg['feature_dim']
        self.feature_rate = self.task_cfg['feature_rate']

        self.is_decoder_finetune = False # TODO bad way of passing in info

    def get_downsample_rates(self, key: str) -> int:
        return 320
        # self.downsample_rate = round(self.sample_rate / self.feature_rate * self.feature_dim/(16*16))

    def wav_to_spectrogram(self, wav):
        return torchaudio.compliance.kaldi.fbank(  # Frame shift and length are standard at  10, 25
            waveform=wav,
            sample_frequency=self.sample_rate,
            use_energy=False,
            num_mel_bins=self.feature_dim
        )

    def forward(self, wavs):
        device = wavs[0].device

        features = [self.wav_to_spectrogram(wav.unsqueeze(0)) for wav in wavs]
        feature_lengths = torch.LongTensor([len(feature) for feature in features]).to(device)
        feature_padding_mask = ~torch.lt(
            torch.arange(max(feature_lengths)).unsqueeze(0).to(device),
            feature_lengths.unsqueeze(1),
        )
        padded_features = pad_sequence(features, batch_first=True)

        results = self.model(padded_features, padding_mask=feature_padding_mask, mask=False, features_only=True, is_decoder_finetune=self.is_decoder_finetune)
        
        return {"last_hidden_state": results["x"], "hidden_states": results["hidden_states"]}
