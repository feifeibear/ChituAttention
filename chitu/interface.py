from flash_attn.flash_attn_interface import _flash_attn_forward

from chitu.int8_flash_attention.flash_atten_int8 import attention_int8
from chitu.int8_flash_attention.flash_atten_full_int8 import attention_full_int8
from chitu.int8_flash_attention.quant import quant_pertoken, quant_pertensor
from chitu.sageattention.core import sageattn

import torch


def _int8_flash_attn_forward(
    query,
    key,
    value,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):

    assert dropout_p == 0.0
    assert softcap == 0.0
    assert alibi_slopes is None
    assert deterministic == False
    assert return_attn_probs == False
    assert window_size == (-1, -1)

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    q8, qs8 = quant_pertoken(query)
    k8, ks8 = quant_pertoken(key)
    v8, vs8 = quant_pertensor(value)

    # Add NaN checks
    assert not torch.isnan(q8).any(), "NaN detected in q8"
    assert not torch.isnan(qs8).any(), "NaN detected in qs8"
    assert not torch.isnan(k8).any(), "NaN detected in k8"
    assert not torch.isnan(ks8).any(), "NaN detected in ks8"
    assert not torch.isnan(v8).any(), "NaN detected in v8"
    assert not torch.isnan(vs8).any(), "NaN detected in vs8"

    if causal:
        out = attention_int8(q8, k8, v, qs8, ks8, causal, softmax_scale)
    else:
        out = attention_full_int8(q8, k8, v8, qs8, ks8, vs8, causal, softmax_scale)

    out = out.transpose(1, 2)
    assert not torch.isnan(out).any(), "NaN detected in output"
    return out


def _sage_attn_forward(
    query,
    key,
    value,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    ret_lse=False,
):
    assert dropout_p == 0.0
    assert softcap == 0.0
    assert alibi_slopes is None
    assert deterministic == False
    assert return_attn_probs == False
    assert window_size == (-1, -1)

    # Convert window_size to attn_mask if needed
    attn_mask = None
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    output = sageattn(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=causal,
        scale=softmax_scale,
        ret_lse = ret_lse
    )
    if ret_lse:
        o, lse = output
        return o.transpose(1, 2), lse
    else:
        return output.transpose(1, 2)
