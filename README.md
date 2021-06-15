# fairseq-to-huggingface
Convert seq2seq models in fairseq (e.g., bart, all-share-embedding transformer) to the format of huggingface-transformers

Most of the codes in convert.py are based on [tomsherborne/example_bart_convert.sh](https://gist.github.com/tomsherborne/e7b629ee9cf0618febb211683a410ce5).

The version of transformers is [v3.5.1](https://github.com/huggingface/transformers/tree/v3.5.1).

## Requirements and Installation

### Transformers
Transformers (modified) version v3.5.1 can be installed as follows:

```
cd transformers
pip install --editable .
```

I modified SinusoidalPositionalEmbedding in transformers/src/transformers/modeling_bart.py to match the implementation in fairseq, since fairseq differs from HuggingFace in sinusoidal embeddings initialization and calculation of positional ids.

#### Why 3.5.1?
Some configurations of BART are fixed in the latest version (>= 4.0.0). For example, Positional Embedding can only choose "learned" instead of "sinusoidal".
In addition, the beam search in the earlier versions has bugs.
Therefore, 3.5.1 is a better choice.

### fairseq
The version of fairseq is 1.0.0a0. The latest version (> 1.0.0) is also ok.
If you want to use it in version 0.9.x or 0.10.x, you need to change args.model.xxx to args.xxx in convert.py, since fairseq adopted the Hydra configuration framework in the latest version.

## Usage

```
# the path of fairseq checkpoint
CHECK_PATH=/to/path/checkpoint*.pt  
# the directory containing dict.src.txt and dict.tgt.txt
DATA_DIR=/to/path/bin_data 
# the directory where config.json and pytorch_model.bin will be saved
OUT_DIR=/to/path/outputs 

python convert.py --checkpoint_path $CHECK_PATH --data_dir $DATA_DIR --save_dir $OUT_DIR
```

```
# the input file
INP=/to/path/*.src
# the path of sentencepiece model
SPM_PATH=/to/path/*.spm
# the path of vocab file
VOCAB=$DATA_DIR/dict.src.txt 
# the path of output file
OUT=*.txt

python pred.py --model_dir $OUT_DIR --input_path $INP --spm_path $SPM_PATH --vocab_path $VOCAB --output_path $OUT

```
