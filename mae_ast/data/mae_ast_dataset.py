# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
import sys
from typing import Any, List, Optional, Union

import numpy as np

import torch
import torch.nn.functional as F
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset

import torchaudio

logger = logging.getLogger(__name__)


def load_audio(manifest_path, max_keep, min_keep):
    n_long, n_short = 0, 0
    names, inds, sizes = [], [], []
    with open(manifest_path) as f:
        root = f.readline().strip()
        for ind, line in enumerate(f):
            items = line.strip().split("\t")
            assert len(items) == 2, line
            sz = int(items[1])
            if min_keep is not None and sz < min_keep:
                n_short += 1
            elif max_keep is not None and sz > max_keep:
                n_long += 1
            else:
                names.append(items[0])
                inds.append(ind)
                sizes.append(sz)
    tot = ind + 1
    logger.info(
        (
            f"max_keep={max_keep}, min_keep={min_keep}, "
            f"loaded {len(names)}, skipped {n_short} short and {n_long} long, "
            f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
        )
    )
    return root, names, inds, tot, sizes


class MAE_AST_Dataset(FairseqDataset):
    def __init__(
            self,
            manifest_path: str,
            sample_rate: float,
            max_keep_sample_size: Optional[int] = None,
            min_keep_sample_size: Optional[int] = None,
            max_sample_size: Optional[int] = None,
            shuffle: bool = True,
            pad_audio: bool = False,
            normalize: bool = False,
            random_crop: bool = False,
            feature_type: str = "wav",
            feature_dim: int = 36,
            deltas: bool = True,
            feature_rate: int = 100,
    ):
        self.audio_root, self.audio_names, inds, tot, self.sizes = load_audio(
            manifest_path, max_keep_sample_size, min_keep_sample_size
        )
        assert feature_type in ['wav', 'spectrogram', 'fbank', 'mfcc'], feature_type
        self.feature_rate = feature_rate
        self.feature_type = feature_type
        self.feature_dim = feature_dim
        self.deltas = deltas
        self.sample_rate = sample_rate
        self.shuffle = shuffle
        self.random_crop = random_crop

        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.pad_audio = pad_audio
        self.normalize = normalize
        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size}"
        )

    def get_audio(self, index):
        import soundfile as sf
        import av

        wav_path = os.path.join(self.audio_root, self.audio_names[index])
        if (wav_path.endswith(".mkv")):
            with av.open(wav_path, metadata_errors="ignore") as container:
                decode = container.decode(audio=0)
                first_frame = next(decode)
                cur_sample_rate = first_frame.sample_rate
                aframes_list = [first_frame.to_ndarray()]
                for frame in decode:
                    aframes_list.append(frame.to_ndarray())
                aframes = np.concatenate(aframes_list, 1)
                wav = torch.as_tensor(aframes).mean(dim=0)
        else:
            wav, cur_sample_rate = sf.read(wav_path)
            wav = torch.from_numpy(wav).float()
        if self.feature_type == "wav":
            feat = self.postprocess_wav(wav, cur_sample_rate)
        else:
            feat = self.postprocess_spec(wav, cur_sample_rate)
        return feat

    def __getitem__(self, index):
        wav = self.get_audio(index)
        return {"id": index, "source": wav}  # , "label_list": labels}

    def __len__(self):
        return len(self.sizes)

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav, 0

        start, end = 0, target_size
        if self.random_crop:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
        return wav[start:end], start

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        audios = [s["source"] for s in samples]
        audio_sizes = [len(s) for s in audios]
        if self.pad_audio:
            audio_size = min(max(audio_sizes), self.max_sample_size)
        else:
            audio_size = min(min(audio_sizes), self.max_sample_size)
        collated_audios, padding_mask, audio_starts = self.collater_audio(
            audios, audio_size
        )

        net_input = {"source": collated_audios, "padding_mask": padding_mask}
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
        }

        return batch

    def collater_audio(self, audios, audio_size):
        if self.feature_type == "wav":
            collated_audios = audios[0].new_zeros(len(audios), audio_size)
        else:
            feat_dim = self.feature_dim * 3 if self.deltas else self.feature_dim
            collated_audios = audios[0].new_zeros(len(audios), audio_size, feat_dim)

        padding_mask = (
            torch.BoolTensor(collated_audios.shape[:2]).fill_(False)
        )
        audio_starts = [0 for _ in audios]
        for i, audio in enumerate(audios):
            diff = len(audio) - audio_size
            if diff == 0:
                collated_audios[i] = audio
            elif diff < 0:
                assert self.pad_audio
                collated_audios[i] = torch.cat([audio, audio.new_full((-diff,), 0.0)])
                padding_mask[i, diff:] = True
            else:
                collated_audios[i], audio_starts[i] = self.crop_to_max_size(
                    audio, audio_size
                )
        return collated_audios, padding_mask, audio_starts

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        if self.pad_audio:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]

    def postprocess_wav(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav

    def postprocess_spec(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            wav = torchaudio.functional.resample(wav, cur_sample_rate, self.sample_rate)

        wav = wav.view(1, -1)
        if self.feature_type == "spectrogram":
            feat = torchaudio.compliance.kaldi.spectrogram(
                waveform=wav,
                sample_frequency=self.sample_rate
            )  # (time, freq)
        elif self.feature_type == "fbank":
            feat = torchaudio.compliance.kaldi.fbank(
                waveform=wav,
                sample_frequency=self.sample_rate,
                use_energy=False,
                num_mel_bins=self.feature_dim
            )  # (time, freq)
        else:
            feat = torchaudio.compliance.kaldi.mfcc(
                waveform=wav,
                sample_frequency=self.sample_rate,
                use_energy=False,
            )  # (time, freq)
        feat = feat[:, :self.feature_dim]
        if self.deltas:
            feat = feat.transpose(0, 1)  # (freq, time)
            deltas = torchaudio.functional.compute_deltas(feat)
            ddeltas = torchaudio.functional.compute_deltas(deltas)
            concat = torch.cat([feat, deltas, ddeltas], dim=0)
            concat = concat.transpose(0, 1).contiguous()
            return concat
        else:
            return feat
