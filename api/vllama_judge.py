from vllm.model_executor.models.llama import LlamaForCausalLM
import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Union, Tuple, List
from transformers.cache_utils import Cache
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import time
import torch
from torch import nn
from transformers import LlamaConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    get_compressed_tensors_cache_scale)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, kv_cache_scales_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.utils import is_hip
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
#from vllm.utils import PPMissingLayer, is_pp_missing_parameter, make_layers

import inspect
from typing import Optional

import torch
import torch.nn as nn

from vllm.distributed import (tensor_model_parallel_all_gather,
                              tensor_model_parallel_gather)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.platforms import current_platform
import math


from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    get_compressed_tensors_cache_scale)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, kv_cache_scales_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.utils import is_hip

from vllm.model_executor.models.interfaces import SupportsLoRA
from vllm.model_executor.models.utils import PPMissingLayer, is_pp_missing_parameter, make_layers



def _prune_hidden_states(
    hidden_states: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    return hidden_states.index_select(0,
                                      sampling_metadata.selected_token_indices)


def _apply_logits_processors(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    found_logits_processors = False
    logits_processed = 0
    for seq_group in sampling_metadata.seq_groups:
        seq_ids = seq_group.seq_ids
        sampling_params = seq_group.sampling_params
        logits_processors = sampling_params.logits_processors
        if logits_processors:
            found_logits_processors = True

            for seq_id, logits_row_idx in zip(seq_ids,
                                              seq_group.sample_indices):
                logits_row = logits[logits_row_idx]
                past_tokens_ids = seq_group.seq_data[seq_id].output_token_ids
                prompt_tokens_ids = seq_group.seq_data[seq_id].prompt_token_ids

                for logits_processor in logits_processors:
                    parameters = inspect.signature(logits_processor).parameters
                    if len(parameters) == 3:
                        logits_row = logits_processor(prompt_tokens_ids,
                                                      past_tokens_ids,
                                                      logits_row)
                    else:
                        logits_row = logits_processor(past_tokens_ids,
                                                      logits_row)

                logits[logits_row_idx] = logits_row

        logits_processed += len(seq_group.sample_indices) + len(
            seq_group.prompt_logprob_indices)

    if found_logits_processors:
        # verifies that no rows in logits were missed unexpectedly
        assert logits_processed == logits.shape[0]
    return logits





class ChoiceLogitsProcessor(nn.Module):
    """Process logits and apply logits processors from sampling metadata.

    This layer does the following:
    1. Gather logits from model hidden_states.
    2. Scale logits if needed.
    3. Apply logits processors (if any).
    """

    def __init__(self,
                 vocab_size: int,
                 org_vocab_size: Optional[int] = None,
                 scale: float = 1.0,
                 logits_as_input: bool = False,
                 soft_cap: Optional[float] = None) -> None:
        """
        Args:
            scale: A scaling factor to apply to the logits.
        """
        super().__init__()
        self.scale = scale
        self.vocab_size = vocab_size
        # Whether the input is logits (default is hidden states).
        self.logits_as_input = logits_as_input
        # original vocabulary size (without LoRA).
        self.org_vocab_size = org_vocab_size or vocab_size
        # Soft cap the logits. Used in Gemma 2.
        self.soft_cap = soft_cap
        # Whether to use gather or all-gather to gather the logits.
        self.use_gather = not current_platform.is_tpu()

    def forward(
        self,
        lm_head: torch.nn.Linear,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        #print("logits_as_input:",self.logits_as_input)
        if self.logits_as_input:
            logits = hidden_states
        else:
            hidden_states = _prune_hidden_states(hidden_states,
                                                 sampling_metadata)

            # Get the logits for the next tokens.
            logits = self._get_logits(hidden_states, lm_head, embedding_bias)
        if logits is not None:
            if self.soft_cap is not None:
                logits = logits / self.soft_cap
                logits = torch.tanh(logits)
                logits = logits * self.soft_cap

            if self.scale != 1.0:
                logits *= self.scale

            # Apply logits processors (if any).
            logits = _apply_logits_processors(logits, sampling_metadata)

            # inf_tensor = torch.full((logits.shape[0], 15), -math.inf, device=logits.device)
            # # 沿着列的维度（dim=1）将 inf_tensor 和原始 tensor 拼接
            # logits = torch.cat((inf_tensor, logits), dim=1)

        return logits

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        # Get the logits for the next tokens.
        # logits = lm_head.linear_method.apply(lm_head,
        #                                      hidden_states,
        #                                      bias=embedding_bias)
        # if self.use_gather:
        #     # None may be returned for rank > 0
        #     logits = tensor_model_parallel_gather(logits)
        # else:
        #     # Gather is not supported for some devices such as TPUs.
        #     # Use all-gather instead.
        #     # NOTE(woosuk): Here, the outputs of every device should not be None
        #     # because XLA requires strict SPMD among all devices. Every device
        #     # should execute the same operations after gathering the logits.
        #     logits = tensor_model_parallel_all_gather(logits)
        # # Remove paddings in vocab (if any).
        logits = lm_head(hidden_states)
        logits = logits.float()

        if logits is not None:
            logits = logits[..., :self.org_vocab_size]
        
        return logits

    def extra_repr(self) -> str:
        s = f"vocab_size={self.vocab_size}"
        s += f", forg_vocab_size={self.org_vocab_size}"
        s += f", scale={self.scale}, logits_as_input={self.logits_as_input}"
        return s




class LlamaForJudge(LlamaForCausalLM):
    def __init__(self, config, cache_config=None, quant_config=None, lora_config=None):
        super().__init__(config, cache_config, quant_config, lora_config)

    def create_optimized_lm_head(self, token_indices: List[int]):
        # 提取指定token对应的权重
        weights = [self.lm_head.weight[idx, :].unsqueeze(0) for idx in token_indices]
        # 形成新的小矩阵
        optimized_weights = torch.cat(weights, dim=0)
        
        # 创建一个新的线性层
        optimized_lm_head = nn.Linear(self.config.hidden_size, len(token_indices), bias=False)
        optimized_lm_head.weight = nn.Parameter(optimized_weights)
        return optimized_lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        model_output = self.model(input_ids, positions, kv_caches,
                                  attn_metadata, intermediate_tensors)
        return model_output

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        
        # logits = self.logits_processor(self.lm_head, hidden_states,
        #                                sampling_metadata)
        logits_processors=sampling_metadata.seq_groups[0].sampling_params.logits_processors
        choice=False
        if logits_processors is not None and len(logits_processors)>0:
            for processor in logits_processors:
                if hasattr(processor, 'is_choice_processor'):
                    choice=True
                    break
        if  choice:
            logits = self.choice_logits_processor(self.optimized_lm_head,hidden_states, sampling_metadata)
        else:
            logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        #print(f"logits_processors:{logits_processors}\nchoice:{choice}\nlogits:{logits[:,:20]}\n")
        return logits
    

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            # With tie_word_embeddings, we can skip lm_head.weight
            # The weight might appear unnecessarily in the files if the model is
            # processed with quantization, LoRA, fine-tuning, etc.
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            if scale_name := get_compressed_tensors_cache_scale(name):
                # Loading kv cache scales for compressed-tensors quantization
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = loaded_weight[0]
                weight_loader(param, loaded_weight)
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)

                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
        token_indices = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        self.optimized_lm_head = self.create_optimized_lm_head(token_indices)
        self.choice_logits_processor = ChoiceLogitsProcessor(self.unpadded_vocab_size,len(token_indices))
