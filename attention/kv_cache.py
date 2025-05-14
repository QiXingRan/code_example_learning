# https://zhuanlan.zhihu.com/p/662498827 KV cache计算

import torch

layer_past = ["存在过KV计算"]
use_cache = True
key = value = 0

# 通过concat将过去+现在的kv拼接起来
if layer_past is not None:
    past_key, past_value = layer_past
    key = torch.cat((past_key, key), dim=-2)
    value = torch.cat((past_value, value), dim=-2)

if use_cache is True:
    present = (key, value)
else:
    present = None

# if self.reorder_and_upcast_attn:
#     attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
# else:
#     attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)


"""
根据attention序列输出的方法，每次输出 Attention_k 仅与当前的 Q_k 相关， 推理当前字符 X_k 只需要输入 X_(k-1)
----> 仅需要根据 Q_k 计算当前 Attention_k， 把每一步的KV缓存下来可以避免重新计算（但是当输入序列过长，容易导致显存占用大）

！！！ 
只能用于Decoder架构的模型，这是因为Decoder有Causal Mask，在推理的时候前面已经生成的字符不需要与后面的字符产生attention，从而使得前面已经计算的K和V可以缓存起来
！！！
"""