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

## Beam search

Beam search in Transfomrers is almost the same as fairseq, but with less effective implementation. Its default configuraion is different from fairseq, e.g., no_repeat_ngram_size, repetition_penalty, length_penalty, num_beams, min_length and early stop.

### What is early stop
When some beams ends (<eos> is generated), Transformers and fairseq both put the sequence into the candidate set.
When the number of candidates is equal to beam size, the generation in fairseq is terminated. While Transformers (early_stop=False) continues to generate tokens, until the score of the new sequence cannot exceed the sentences in the candidate set. If we set early_stop=True, it can be consistent with fairseq.

### Related codes
```
from tqdm import tqdm
data_size = len(data_lines)
batch_size = 8
max_length = 200
trans_all_results = []
# 以下为超参数设置
transformers_bart.config.no_repeat_ngram_size = None
transformers_bart.config.repetition_penalty = None
transformers_bart.config.length_penalty = 1.0
transformers_bart.config.num_beams = 5
transformers_bart.config.min_length = None
for start_idx in tqdm(range(0, data_size, batch_size)):
    batch_lines = [line for line in data_lines[start_idx: min(start_idx + batch_size, data_size)]]
    inp = transformer_tokenizer.prepare_seq2seq_batch(batch_lines, return_tensors='pt')
    summaries = transformers_bart.generate(inp['input_ids'][:, 1:].to(device),  # 因为fairseq-bart-gec训练时，没有在开头添加<s>，所以这里把第一个id去除
                attention_mask=inp['attention_mask'][:, 1:].to(device),  # 同上
                num_beams=5,
                max_length=max_length + 2,
                early_stopping=True,  # 使得和fairseq保持一致
                decoder_start_token_id=2,  # 因为fairseq-bart-gec训练时，decoder端第一个输入为</s>，所以这里设置为</s>的id -> 2
            )
trans_all_results.extend(transformer_tokenizer._decode(hypos, skip_special_tokens=True,  clean_up_tokenization_spaces=False) for hypos in summaries)  # 这里clean_up_tokenization_spaces=False是为了与fairseq保持一致，指不对文本做后处理；后处理指的是，如果生成的是“end .”（标点前含空格），会把它改为“end.”（标点前不含空格）。也就是说fairseq默认只依靠bart来判断是否加空格，如果bart预测错了，并不用规则更正
```
