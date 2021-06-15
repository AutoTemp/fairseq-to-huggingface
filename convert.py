from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BartForSequenceClassification,
    BartModel,
    BartTokenizer,
)
from fairseq.checkpoint_utils import load_model_ensemble_and_task
import os
import logging
import sys
from pathlib import Path
import argparse
import torch

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s |  [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("convert")

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, required=True)
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--save_dir", type=str, required=True)
main_args = parser.parse_args()

logger.info('Load fairseq checkpoint...')
models, args, task = load_model_ensemble_and_task(filenames=[os.path.expanduser(main_args.checkpoint_path)],
                                                 arg_overrides={'data': os.path.expanduser(main_args.data_dir)})

fairseq_transformer = models[0].eval()

logger.info('Repr from faiseq...')
inp_tokens = torch.tensor([[5834, 422, 2401, 9, 1588, 23, 4, 6174, 1100, 59, 646, 8, \
          675,   34,   29,   10, 2196,   29,    4, 6174, 1100,  223, 4918,    7, \
           47,  229, 1379,    8, 2405, 1976, 1340,    5,    2]])
prev_output_tokens = inp_tokens.clone()
prev_output_tokens[:, 0] = inp_tokens.gather(1, (inp_tokens.ne(task.source_dictionary.pad()).sum(dim=1) - 1).unsqueeze(-1),).squeeze()

prev_output_tokens[:, 1:] = inp_tokens[:, :-1]
features, extra = fairseq_transformer(src_tokens=inp_tokens,
                                      src_lengths=None,
                                      prev_output_tokens=prev_output_tokens,
                                      features_only=True,
                                      return_all_hiddens=False,
                                      )
logger.info('Huggingface config...')
huggingface_config = BartConfig.from_pretrained('facebook/bart-large',
                                                activation_function=args.model.activation_fn,
                                                d_model=args.model.encoder_embed_dim,
                                                encoder_attention_heads=args.model.encoder_attention_heads, 
                                                encoder_ffn_dim=args.model.encoder_ffn_embed_dim, 
                                                encoder_layers=args.model.encoder_layers,
                                                decoder_attention_heads=args.model.decoder_attention_heads, 
                                                decoder_ffn_dim=args.model.decoder_ffn_embed_dim, 
                                                decoder_layers=args.model.decoder_layers,
                                                normalize_embedding=args.model.layernorm_embedding, 
                                                scale_embedding=(not args.model.no_scale_embedding), 
                                                static_position_embeddings=(not args.model.encoder_learned_pos),
                                                vocab_size=len(task.source_dictionary)
                                               )
logger.info('Init huggingface model...')
huggingface_model = BartForConditionalGeneration(huggingface_config).eval()

logger.info('Convert...')
def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
        "decoder.output_projection.weight",
        "encoder.embed_positions._float_tensor",
        "decoder.embed_positions._float_tensor"
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)

def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val
    
state_dict = fairseq_transformer.state_dict()
remove_ignore_keys_(state_dict)
state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
huggingface_model.model.load_state_dict(state_dict, strict=False)

logger.info('Repr from huggingface...')
huggingface_features = huggingface_model.model(inp_tokens)[0]
logger.info(huggingface_features.size())
logger.info(features.size())

if huggingface_features.eq(features).all():
    assert features.shape == huggingface_features.shape
    assert (features == huggingface_features).all().item()
    logger.info('Success!')
    Path(main_args.save_dir).mkdir(exist_ok=True)
    huggingface_model.save_pretrained(main_args.save_dir)        
else:
    assert False, 'Wrong'                                           






