import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, LlamaConfig
from peft import LoraConfig, get_peft_model
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import LlamaForCausalLM,LlamaModel,LlamaDecoderLayer
from typing import List, Optional, Tuple, Union
import torch.utils.checkpoint
from transformers.utils import (
    logging,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

logger = logging.get_logger(__name__)

class MyLlamaConfig(LlamaConfig):
    # 这里添加你的自定义配置参数
    def __init__(self, your_custom_param=None, **kwargs):
        super().__init__(**kwargs)
        self.your_custom_param = your_custom_param

class MyLlamaModel(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        
    def forward(
        self,
        starting_layer: int = 0,
        ending_layer: int = -1,
        ckpt_input = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        if ckpt_input == None:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

            if self.gradient_checkpointing and self.training and use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
                )
                use_cache = False

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)

            if use_cache and past_key_values is None:
                past_key_values = DynamicCache()

            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )

            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)

            causal_mask = self._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
            )

            hidden_states = inputs_embeds

            # create position embeddings to be shared across the decoder layers
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        else:
            hidden_states = ckpt_input["hidden_states"]
            causal_mask = ckpt_input["attention_mask"]
            position_ids = ckpt_input["position_ids"]
            past_key_values = ckpt_input["past_key_values"]
            output_attentions = ckpt_input["output_attentions"]
            use_cache = ckpt_input["use_cache"]
            cache_position = ckpt_input["cache_position"]
            position_embeddings = ckpt_input["position_embeddings"]

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[starting_layer : ending_layer]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if ending_layer == -1:
            hidden_states = self.norm(hidden_states)
        else:
            ckpt_output = {
                "hidden_states": hidden_states,
                "attention_mask": causal_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "output_attentions": output_attentions,
                "use_cache": use_cache,
                "cache_position": cache_position,
                "position_embeddings": position_embeddings
            }
            return ckpt_output

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

class MyLlamaForCausalLM(LlamaForCausalLM):
    config_class = MyLlamaConfig
    def __init__(self, config):
        super().__init__(config)
        self.model = MyLlamaModel(config)

    def forward(
        self,
        starting_layer: int = 0,
        ending_layer: int = -1,
        ckpt_input = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            starting_layer = starting_layer,
            ending_layer = ending_layer,
            ckpt_input = ckpt_input,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        if not ending_layer == -1:
            return outputs

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class Classifier(nn.Module):
    def __init__(self, cls_num = 2):
        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(4096, 1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, cls_num)

    def forward(self, input):#b,4096

        output = self.fc1(input)
        output = self.relu1(output)
        output = self.dropout1(output)
        output = self.fc2(output)
        output = self.relu2(output)
        output = self.dropout2(output)
        output = self.fc3(output)

        return output

class RecurrentLlama(nn.Module):
    def __init__(self, N=22, M=33, num_loops=3, num_classes=2, device='cuda',model_id=None,lora_rank=8):
        super().__init__()
        self.device = device
        self.N = N  # 从N层微调
        self.M = M  # 到后M层
        self.num_loops = num_loops  # 循环次数

        # AutoModelForCausalLM.register(MyLlamaConfig, MyLlamaForCausalLM)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = "[PAD]"
        self.tokenizer.padding_side = "right"

        config_kwargs = {
            "trust_remote_code": True,
            "cache_dir": None,
            "revision": 'main',
            "use_auth_token": None,
            "return_dict_in_generate": True,
            "output_hidden_states": True
        }
        model_config = MyLlamaConfig.from_pretrained(model_id, **config_kwargs)
        self.model = MyLlamaForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            device_map=device,
            torch_dtype=torch.float32)
        self.model.eval()
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank*2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
            lora_dropout=0.1,
        )
        #self.inject_lora(lora_config, starting_layer=N, ending_layer=M)
        for name, param in self.model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        self.config = self.model.model.config
        self.output_head = Classifier(cls_num = num_classes)
        self.translayer = LlamaDecoderLayer(config=self.config,layer_idx=40)

    def inject_lora(self, lora_config, starting_layer=21, ending_layer=32):
        self.lora_config = lora_config
        
        # 直接替换self.layers中的层对象
        # for i in range(starting_layer, ending_layer):
        #     self.model.model.layers[i] = inject_adapter_in_model(
        #         self.model.model.layers[i], 
        #         self.lora_config
        #     )
        for i in range(starting_layer, ending_layer):
            self.model.model.layers[i] = get_peft_model(
                self.model.model.layers[i], 
                self.lora_config, 
                adapter_name="lora"
            )
    
    def save_model(self, save_dir, model_name, epoch):
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存所有LoRA适配器参数
        lora_state_dict = {name: param.cpu() for name, param in self.model.named_parameters() if 'lora' in name}
        # 保存分类头参数
        trans_layer_state = self.translayer.state_dict()
        output_head_state = self.output_head.state_dict()
        
        # 合并保存到单个文件
        torch.save({
            'lora': lora_state_dict,
            'output_head': output_head_state,
            'trans_layer': trans_layer_state,
            'epoch': epoch
        }, os.path.join(save_dir, model_name))

    def load_adapter_head(self, load_path, model_name):
        # 加载保存的检查点
        checkpoint = torch.load(os.path.join(load_path, model_name), map_location=self.device, weights_only=True)
        
        # 加载LoRA参数到模型
        self.model.load_state_dict(checkpoint['lora'], strict=False)

        # 加载转换层参数
        self.translayer.load_state_dict(checkpoint['trans_layer'])
        self.translayer.to(self.device)
        
        # 加载分类头参数
        self.output_head.load_state_dict(checkpoint['output_head'])
        self.output_head.to(self.device)

        return checkpoint['epoch']
    
    def load_front(self):#加载模型的前N层到显存，并将后面的模型参数移动到CPU
        self.model.model.embed_tokens.to('cuda:0')
        self.model.model.norm.to('cuda:0')
        self.model.model.layers[:self.N].to('cuda:0')
        self.model.model.layers[self.N:self.M].to('cpu')
        self.model.model.layers[self.M:].to('cpu')
        self.translayer.to('cpu')
        self.output_head.to('cpu')

    def load_back(self):#加载模型的后M层到显存，并将前面的模型参数移动到CPU
        self.model.model.layers[:self.N].to('cpu')
        self.model.model.layers[self.M:].to('cpu')
        self.model.model.layers[self.N:self.M].to('cuda:0')
        self.translayer.to('cuda:0')
        self.output_head.to('cuda:0')

    def forward(self, sents_batch, max_len, device):

        sents_batch_encoding = self.tokenizer(sents_batch, return_tensors='pt', max_length=max_len, padding="max_length", truncation=True)
        sents_batch_encoding = sents_batch_encoding.to(device)
        with torch.no_grad():
            out_ckpt = self.model.forward(
                starting_layer = 0,
                ending_layer = self.N,
                ckpt_input = None,
                **sents_batch_encoding,
            )
        
        all_logits = []
        
        for i in range(self.num_loops):

            hidden_states_copy = out_ckpt["hidden_states"]

            out_ckpt = self.model.forward(
                starting_layer = self.N,
                ending_layer = self.M,
                ckpt_input = out_ckpt,
            )
            hidden_states = out_ckpt["hidden_states"]

            out_h = torch.mean(hidden_states, dim=1)
            logits = self.output_head(out_h)
            all_logits.append(logits)
            if i == self.num_loops - 1:
                break
            res = self.translayer(
                hidden_states=hidden_states,
                attention_mask=out_ckpt["attention_mask"],
                position_ids=out_ckpt["position_ids"],
                past_key_value=out_ckpt["past_key_values"],
                output_attentions=out_ckpt["output_attentions"],
                use_cache=out_ckpt["use_cache"],
                cache_position=out_ckpt["cache_position"],
                position_embeddings=out_ckpt["position_embeddings"],
            )[0]

            out_ckpt["hidden_states"] = res + hidden_states_copy
        output_dict = {
            "logits" : all_logits
        }
            
        return all_logits
