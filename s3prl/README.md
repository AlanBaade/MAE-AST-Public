# Fine Tuining MAE-AST on Superb with S3prl

This folder (mae_ast in s3prl) is the upstream module for fine-tuning MAE-AST on Superb with S3prl, and acts as a demo for how to use the MAE-AST in downstream applications.

## Setting up a S3prl environment
Here's an example of the environment setup used, given that fairseq and s3prl are cloned as separate folders in the home directory.

```
conda create -n s3prl_mae_ast python=3.7
conda activate s3prl_mae_ast
conda install pathlib
cd ~/s3prl
pip install -e ./
cd ~/fairseq
pip install -e ./
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Rerunning this generally fixes further environment errors
```
cd ~/s3prl
pip install -e ./
```

## Fine-tuning

Fine-tuning commands should be performed in ``~/s3prl/s3prl`` with ``s3prl_mae_ast`` activated
All downstream configuration files used in the paper can be accessed in s3prl/config. These contain default settings or slight modifications to gradient accumulate steps and batch size to mimic default settings on our hardware.

What follows are the exact run commands used alongside these config files for fine-tuning (disregarding template paths):
#### Command to sanity check imports
```
python run_downstream.py -m train -n S3prl_mae_ast_sanity -u wav2vec2 -d example -f
```

#### Keyword Spotting 1 (KS1) (Speech Commands)
```
python run_downstream.py -m train -n MAE_AST_Template_Name -u mae_ast -d speech_commands -f \
  -k /path/to/checkpoint.pt \
  -s last_hidden_state \
  -o "config.downstream_expert.datarc.speech_commands_root='/path/to/speech_commands_v0.01/',,\
  config.downstream_expert.datarc.speech_commands_test_root='/path/to/speech_commands_test_set_v0.01/',,\
  config.optimizer.lr=1.0e-5"
```

#### Speaker Identification (SID) (VoxCeleb1)
```
python run_downstream.py -m train -n MAE_AST_Template_Name -u mae_alan -d voxceleb1 -f \
  -k /path/to/checkpoint.pt \
  -s hidden_states \
  -o "config.downstream_expert.datarc.file_path='/path/to/VoxCeleb1/',,\
  config.optimizer.lr=1.0e-4"
```

#### Emotion Recognition IEMOCAP (ER)
Recall ER takes place over five folds, with the resulting test score being the average of the tests from each fold.
```
for test_fold in fold1 fold2 fold3 fold4 fold5;
do
python run_downstream.py -m train -n MAE_AST_Template_Name$test_fold -u mae_alan -d emotion -f \
  -k /path/to/checkpoint.pt \
  -s last_hidden_state \
  -o "/path/to/IEMOCAP_full_release',,\
  config.downstream_expert.datarc.test_fold=$test_fold,,\
  config.optimizer.lr=1.0e-4"
done
```
