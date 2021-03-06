# SPINN augmented with Lifted Matrix-Space model

This repository contains the source code based on the paper [The Lifted-Matrix Space Model for Semantic Composition][1]. The Lifted Matrix-Space (LMS) model was implemented on top of the [SPINN Python Codebase][2].


The included implementations are:

- A **Python/Pytorch** implementation of LMS and SPINN, using a naïve stack representation (named `fat-stack`)

## Python code

The Python code lives, quite intuitively, in the `python` folder.

### Installation

Requirements:

- Python 2.7
- Pytorch

Install most required Python dependencies using the command below.

    pip install -r python/requirements.txt

Install Pytorch based on instructions online: http://pytorch.org

### Running the code

The main executable for the SNLI experiments in the paper is [supervised_classifier.py](https://github.com/woojinchung/lms/blob/master/python/spinn/models/supervised_classifier.py), whose flags specify the hyperparameters of the model. You can specify gpu usage by setting `--gpu` flag greater than or equal to 0. Uses the CPU by default.

Here's a sample command that runs a fast, low-dimensional CPU training run, training and testing only on the dev set. It assumes that you have a copy of [SNLI](http://nlp.stanford.edu/projects/snli/) available locally.

    PYTHONPATH=spinn/python \
        python2.7 -m spinn.models.supervised_classifier --data_type snli \
        --training_data_path ~/data/snli_1.0/snli_1.0_dev.jsonl \
        --eval_data_path ~/data/snli_1.0/snli_1.0_dev.jsonl \
        --embedding_data_path python/spinn/tests/test_embedding_matrix.5d.txt \
        --word_embedding_dim 5 --model_dim 10 --model_type CBOW

For full runs, you'll also need a copy of the 840B word 300D [GloVe word vectors](http://nlp.stanford.edu/projects/glove/).

## Log Analysis

This project contains a handful of tools for easier analysis of your model's performance.

For one, after a periodic number of batches, some useful statistics are printed to a file specified by `--log_path`. This is convenient for visual inspection, and the script [parse_logs.py](https://github.com/mrdrozdov/spinn/blob/master/scripts/parse_logs.py) is an example of how to easily parse this log file.

In addition, there is support for realtime summaries using [Visdom](https://github.com/facebookresearch/visdom). This requires a few steps:

1. Run your experiment normally, but specify a `--metrics_path`.
2. Run Visdom in it's own terminal instance: `python -m visdom.server`
3. Run this project's [visdom_reporter.py](https://github.com/mrdrozdov/spinn/blob/master/scripts/visdom_reporter.py) script, specifying a root which matches the `--metrics_path` flag: `python scripts/visdom_reporter.py --root $METRICS_PATH`

Then open Visdom in a browser window to see graphs representing accuracy, loss and some other metrics updated in real time. This is most useful when running multiple experiments simultaneously.

## License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use these files except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

[1]: https://arxiv.org/abs/1711.03602
[2]: https://github.com/nyu-mll/spinn
