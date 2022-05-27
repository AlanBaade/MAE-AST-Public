# MAE-AST
This repository contains the code for the paper [MAE-AST: Masked Autoencoding Audio Spectrogram Transformer](https://arxiv.org/abs/2203.16691). Pretrained checkpoints to be hosted in the coming few days.

This repository contains three folders: config, mae_ast, and s3prl. Config contains a default pre-training config for the mae-ast. The mae_ast folder contains the main code for the model, and runs under [fairseq](https://github.com/facebookresearch/fairseq). This includes a criterion, task, data loading, and models. The s3prl folder provides the upstream model and configuration for fine-tuning the MAE-AST on Superb tasks under the [S3prl repository](https://github.com/s3prl/s3prl). This repository does not include fine-tuning code for AudioSet, Librispeech, and KS2, which are instead evaluated under the [SSAST library](https://github.com/YuanGongND/ssast) with no settings changed.

Please email abaade@utexas.edu for questions.

## Pretrained Model Download
Below are the two 12-layer models used in the overall results section of the paper, with a masking ratio of 75%. Clicking the link attempts to display the model checkpoints as a text file. Use wget or open the link in a new tab and save.
| Download | Model         | Layers | Masking | AS    | ESC-50 | KS2   | KS1   | SID   | ER    |
|-----------|---------------|--------|---------|-------|--------|-------|-------|-------|-------|
| [Checkpoint](https://saltlab.cs.utexas.edu/downloads/model_checkpoints/mae_ast/chunk_patch_75_12LayerEncoder.pt)         | MAE-AST Patch |     12 | Chunked | 0.306 |  0.900 | 0.979 | 0.958 | -     | 0.598 |
| [Checkpoint](https://saltlab.cs.utexas.edu/downloads/model_checkpoints/mae_ast/random_frame_75_12LayerEncoder.pt)         | MAE-AST Frame |     12 | Random  | 0.230 |  0.889 | 0.980 | 0.973 | 0.633 | 0.621 |

## Pre-Training
Pretraining on fairseq is done as follows

### Environment Setup
Run the following commands with conda to set up an environment for pretraining. This assumes that fairseq is downloaded to the home directory
```
conda create -n fairseq_mae_ast python=3.9
conda activate fairseq_mae_ast
pip install soundfile
cd ~/fairseq
pip install -e ./
conda install tensorboardX
conda install av -c conda-forge
pip install sortedcontainers
pip install tensorboard
```

### Input files
The dataset code takes in a directory which contains the files train.tsv, valid.tsv, and test.tsv, containing paths to the train, valid, and test data respectively. Each of train.tsv, valid.tsv, and test.tsv are tab separated value files with a ``/`` on the first line, followed by lines with (audio file paths, tab, length in frames of that audio file). For example, train.tsv starts with:
```
/
/path/to/AudioSet/unbalanced/6XUF56FlKvg.mkv     479232
/path/to/data/AudioSet/unbalanced/eJS_911G6ps.mkv     477696
```
and test.tsv starts with:
```
/
/path/to/LibriSpeech/data/test-other/3331/159609/3331-159609-0002.flac       225600
/path/to/LibriSpeech/data/test-other/3331/159609/3331-159609-0021.flac       165920
```
The dataset expects either mkv or flac files as input.

### Environment Variables
Let MAE-AST-Public be the base directory of this repository

Run the following to set up enviroment variables
```
conda activate fairseq_mae_ast
cd ~/MAE-AST-Public
export HYDRA_FULL_ERROR=1
data_dir=/path/to/directory_with_train_valid_test_tsv_input_files
config_dir=/path/to/MAE-AST-Public/config/pretrain
user_dir=/path/to/MAE-AST-Public/mae_ast
```

### Pretraining commands
The following run commands overwrite the default pretrain configuration, and contain the most important settings to change.

The code for configuration settings is at the top of ``mae_ast/models/mae_ast.py`` and ``mae_ast/tasks/mae_ast_pretraining.py``. The main model logic (model forward pass) is in the middle of ``mae_ast/models/mae_ast.py``

#### Patched, Chunked Masking (SSAST), 12 Layer Encoder, 75% masking ratio
Default Model Patch (12 Layer).
```
fairseq-hydra-train \
  --config-dir ${config_dir} --config-name mae_ast common.user_dir=${user_dir} task.data=${data_dir} model._name=mae_ast criterion._name=mae_ast \
  model.encoder_layers=12 model.decoder_layers=2 \
  model.random_mask_prob=0.75 task.mask_type="chunk_mask" \
  model.ast_kernel_size_chan=16 model.ast_kernel_size_time=16 model.ast_kernel_stride_chan=16 model.ast_kernel_stride_time=16 \
  criterion.classification_weight=1 criterion.reconstruction_weight=10 \
  distributed_training.distributed_world_size=1 distributed_training.nprocs_per_node=1 \
  common.log_interval=200 checkpoint.save_interval_updates=25000 \
  optimization.max_update=550000 dataset.max_tokens=8388608 optimization.lr=[0.0001]\
  hydra.run.dir=/path/to/output_model_directory
```

#### Frame, Random Masking, 12 Layer Encoder, 75% masking ratio
Default Model Frame (12 Layer).
Changing the kernel sizes and strides determines frame vs patch models.
```
fairseq-hydra-train \
  --config-dir ${config_dir} --config-name mae_ast common.user_dir=${user_dir} task.data=${data_dir} model._name=mae_ast criterion._name=mae_ast \
  model.encoder_layers=12 model.decoder_layers=2 \
  model.random_mask_prob=0.75 task.mask_type="random_mask" \
  model.ast_kernel_size_chan=128 model.ast_kernel_size_time=2 model.ast_kernel_stride_chan=128 model.ast_kernel_stride_time=2 \
  criterion.classification_weight=1 criterion.reconstruction_weight=10 \
  distributed_training.distributed_world_size=1 distributed_training.nprocs_per_node=1 \
  common.log_interval=200 checkpoint.save_interval_updates=25000 \
  optimization.max_update=550000 dataset.max_tokens=8388608 optimization.lr=[0.0001]\
  hydra.run.dir=/path/to/output_model_directory
```

#### Frame, Chunked Masking (Wav2Vec2), 12 Layer Encoder, 75% masking ratio
The random mask probability is 1.45 due to overlap in Wav2Vec2-style masking (specified by task.mask_type="retain_spans"), which creates an average 75% masking ratio.
Set the random mask probability to 0.74 for an average of 50% masking. For all other mask types, the random mask probability directly corresponds to the amount of tokens masked.
```
fairseq-hydra-train \
  --config-dir ${config_dir} --config-name mae_ast common.user_dir=${user_dir} task.data=${data_dir} model._name=mae_ast criterion._name=mae_ast \
  model.encoder_layers=12 model.decoder_layers=2 \
  model.random_mask_prob=1.45 task.mask_type="retain_spans" \
  model.ast_kernel_size_chan=128 model.ast_kernel_size_time=2 model.ast_kernel_stride_chan=128 model.ast_kernel_stride_time=2 \
  criterion.classification_weight=1 criterion.reconstruction_weight=10 \
  distributed_training.distributed_world_size=1 distributed_training.nprocs_per_node=1 \
  common.log_interval=200 checkpoint.save_interval_updates=25000 \
  optimization.max_update=550000 dataset.max_tokens=8388608 optimization.lr=[0.0001]\
  hydra.run.dir=/path/to/output_model_directory
```

## Fine-Tuning
The s3prl directory contains an example for fine-tuning the MAE-AST on superb, plus a readme with specific fine-tuning settings. s3prl/mae_ast/hubconf.py takes in a checkpoint generated during pretraining and uses it on downstream tasks.
