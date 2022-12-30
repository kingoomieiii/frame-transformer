from transformers import AutoModel, BertModel
from frame_transformer_v3 import FrameTransformer
import torch

import argparse
import json
import os

def load_from_huggingface(transformer, name='bert-large-uncased'):
    pre = AutoModel.from_pretrained(name)
    
    idx = 0
    for layer in pre.encoder.layer:
        if idx < len(transformer):
            channels = transformer[idx].conv1.weight_pw.shape[0]
            transformer[idx].attn.q_proj.weight_pw.data = expand_weight(layer.attention.self.query.weight, channels)
            transformer[idx].attn.q_proj.bias_pw.data = expand_linear_bias(layer.attention.self.query.bias, channels)
            transformer[idx].attn.k_proj.weight_pw.data = expand_weight(layer.attention.self.key.weight, channels)
            transformer[idx].attn.k_proj.bias_pw.data = expand_linear_bias(layer.attention.self.key.bias, channels)
            transformer[idx].attn.v_proj.weight_pw.data = expand_weight(layer.attention.self.value.weight, channels)
            transformer[idx].attn.v_proj.bias_pw.data = expand_linear_bias(layer.attention.self.value.bias, channels)
            transformer[idx].attn.out_proj.weight_pw.data = expand_weight(layer.attention.output.dense.weight, channels)
            transformer[idx].attn.out_proj.bias_pw.data = expand_linear_bias(layer.attention.output.dense.bias, channels)
            transformer[idx].norm1.weight.data = expand_weight(layer.attention.output.LayerNorm.weight, channels)
            transformer[idx].norm1.bias.data = expand_normal_bias(layer.attention.output.LayerNorm.bias, channels)
            transformer[idx].conv1.weight_pw.data = expand_weight(layer.intermediate.dense.weight, channels)
            transformer[idx].conv1.bias_pw.data = expand_linear_bias(layer.intermediate.dense.bias, channels)
            transformer[idx].conv2.weight_pw.data = expand_weight(layer.output.dense.weight, channels)
            transformer[idx].conv2.bias_pw.data = expand_linear_bias(layer.output.dense.bias, channels)
            transformer[idx].norm2.weight.data = expand_weight(layer.output.LayerNorm.weight, channels)
            transformer[idx].norm2.bias.data = expand_normal_bias(layer.output.LayerNorm.bias, channels)
            idx += 1

    return transformer

def expand_weight(x, channels):
    return x.unsqueeze(0).expand((channels, -1, -1))

def expand_linear_bias(x, channels):
    return x.unsqueeze(0).unsqueeze(-1).expand((channels, -1, -1))

def expand_normal_bias(x, channels):
    return x.unsqueeze(0).unsqueeze(0).expand((channels, -1, -1))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', type=str, default='bert-large-uncased')
    p.add_argument('-n_fft', type=int, default=2048)
    p.add_argument('--num_layers', type=int, default=24)
    p.add_argument('--channels', type=int, default=2)
    p.add_argument('--expansion', type=int, default=4)
    p.add_argument('--num_heads', type=int, default=16)
    p.add_argument('--output', type=str, default='H://models')
    args = p.parse_args()

    outdir = os.path.join(args.output, '')
    model_path = os.path.join(outdir, f'{args.name}-c.{args.channels}-l.{args.num_layers}-ff.{args.expansion}-nh.{args.num_heads}.model.pth')

    model = FrameTransformer(channels=args.channels, n_fft=args.n_fft, num_heads=args.num_heads, expansion=args.expansion, num_layers=args.num_layers)
    print(model.transformer[0].conv1.weight_pw)
    model.transformer = load_from_huggingface(model.transformer)
    torch.save(model.state_dict(), model_path)
    print(model.transformer[0].conv1.weight_pw)

if __name__ == '__main__':
    main()