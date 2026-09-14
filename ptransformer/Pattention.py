import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.modules import RMSNorm
from fla.modules.activations import swiglu
class Pattention(nn.Module):
    """
    Pattention Layer.
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        param_token_num,
        n_layers=12,  # 添加默认层数
        init_std=0.02,  # 添加normal初始化的标准差
        key_init_type="normal",  # 添加key初始化类型，可选"normal"或"wang_init"
        value_init_type="wang_init",  # 添加value初始化类型，可选"normal"或"wang_init"
    ):
        super().__init__()

        self.param_token_num = param_token_num
        self.param_key_dim = input_channels
        self.param_value_dim = output_channels
        self.norm_activation_type = "rmsnorm_sigmoid"
        self.key_init_type = key_init_type
        self.value_init_type = value_init_type
        self.norm = RMSNorm(param_token_num, eps=1e-6)
        self.init_std = init_std
        
        self.key_param_tokens = nn.parameter.Parameter(
            data=torch.rand((self.param_token_num, self.param_key_dim)))
        self.value_param_tokens = nn.parameter.Parameter(
            data=torch.rand((self.param_token_num, self.param_value_dim)))

        self.n_layers = n_layers
        
        # 初始化key参数
        if key_init_type == "normal":
            self._normal_init(self.key_param_tokens, std=init_std)
        elif key_init_type == "wang_init":
            self._wang_init(self.key_param_tokens, n_layers=n_layers, dim=input_channels)
        else:
            raise ValueError(f"不支持的key初始化类型: {key_init_type}")
        
        # 初始化value参数
        if value_init_type == "normal":
            self._normal_init(self.value_param_tokens, std=init_std)
        elif value_init_type == "wang_init":
            self._wang_init(self.value_param_tokens, n_layers=n_layers, dim=output_channels)
        else:
            raise ValueError(f"不支持的value初始化类型: {value_init_type}")
            

    def _normal_init(self, tensor, std=0.02):
        """正态分布初始化方法"""
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)
    
    def _wang_init(self, tensor, n_layers, dim, mup_init_scale=1.0):
        """Wang初始化方法"""
        std = 2 / (n_layers + 1) / math.sqrt(dim)
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    def reset_parameters(self):
        # key
        if self.key_init_type == "normal":
            nn.init.normal_(self.key_param_tokens, mean=0.0, std=self.init_std)
        elif self.key_init_type == "wang_init":
            self._wang_init(self.key_param_tokens,
                            n_layers=self.n_layers,
                            dim=self.param_key_dim)
        else:
            raise ValueError(f"unknown key_init_type {self.key_init_type}")

        # value
        if self.value_init_type == "normal":
            nn.init.normal_(self.value_param_tokens, mean=0.0, std=self.init_std)
        elif self.value_init_type == "wang_init":
            self._wang_init(self.value_param_tokens,
                            n_layers=self.n_layers,
                            dim=self.param_value_dim)
        else:
            raise ValueError(f"unknown value_init_type {self.value_init_type}")
    
    def nonlinear_norm_func(self, inputs, normalize_type, dim=-1):
        if normalize_type == 'softmax': 
            # NOTE: softmax = exp_l1_norm
            # outputs = F.softmax(inputs, dim=dim) * inputs.shape[dim]
            # nonlinear_outputs = torch.exp(inputs)
            # norm_outputs = nonlinear_outputs / torch.norm(nonlinear_outputs, p=1, dim=dim, keepdim=True) * inputs.shape[dim]
            outputs = F.softmax(inputs, dim=dim)
        elif normalize_type == 'gelu_l2_norm':
            nonlinear = F.gelu(inputs)                       # 1) 非线性
            # 2) 先转 fp32 计算范数再转回来，防止 BF16 精度丢失
            l2 = torch.norm(nonlinear, p=2, dim=dim, keepdim=True)
            nonlinear = nonlinear / (l2 + 1e-8).to(nonlinear.dtype)  # 3) 加 eps
            nonlinear *= math.sqrt(nonlinear.shape[dim])     # 4) 同量纲缩放
            outputs = nonlinear
        elif normalize_type == "l2_norm_gelu":
            inputs_float = inputs
            norm = torch.norm(inputs_float, p=2, dim=dim, keepdim=True)
            # 使用 JAX 的方式：在分母加 epsilon (值也用 1e-5)
            norm_outputs = inputs / norm * math.sqrt(inputs.shape[dim])
            outputs = F.gelu(norm_outputs)

        elif normalize_type == "rmsnorm_sigmoid":
            norm = self.norm(inputs)
            # 使用 JAX 的方式：在分母加 epsilon (值也用 1e-5)
            norm_outputs = inputs / norm 
            outputs = torch.sigmoid(norm_outputs) - 0.5

        elif normalize_type == "rmsnorm_gelu":
            norm_outputs = self.norm(inputs)
            outputs = F.gelu(norm_outputs)

        else:
            raise NotImplementedError
        return outputs

    def forward(self, inputs, dropout_p=0.0, router_index=None, attn_mask=None, scale=None):
        query = inputs
        if router_index is None:
            # not MoE mode
            key, value = self.key_param_tokens, self.value_param_tokens
        else:
            key, value = self.key_param_tokens[router_index], self.value_param_tokens[router_index]
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 if scale is None else scale 
        # just for gelu nonlinear, set torch.zeros for softmax
        attn_bias = torch.ones(L, S, dtype=query.dtype, device=query.device)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                # just for gelu nonlinear, set -inf for softmax
                attn_bias.masked_fill_(attn_mask.logical_not(), 0)
            else:
                raise NotImplementedError

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        # just for gelu nonlinear, set attn_weight += attn_bias for softmax
        attn_weight *= attn_bias
        # modified softmax
        attn_weight = self.nonlinear_norm_func(attn_weight, self.norm_activation_type, dim=-1)
        output = attn_weight @ value

        return output

'''class EquivalentMLP(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        hidden_channels, 
        eps=1e-5  # Changed to 1e-5 to match JAX style
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.eps = eps
        
        self.linear1 = nn.Linear(input_channels, hidden_channels, bias=False)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hidden_channels, output_channels, bias=False)

    def l2_norm(self, x, dim=-1):
        norm = torch.norm(x, p=2, dim=dim, keepdim=True)
        norm_outputs = x / norm * math.sqrt(x.shape[dim])
        return norm_outputs

    def forward(self, inputs):
        hidden = self.linear1(inputs)
        normalized_hidden = self.l2_norm(hidden, dim=-1)
        activated_hidden = self.activation(normalized_hidden)
        output = self.linear2(activated_hidden)
        return output'''

class EquivalentMLP(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        hidden_channels,
        vertical_channels, 
        eps=1e-5  # Changed to 1e-5 to match JAX style
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.eps = eps

        self.nextus0 = nn.Linear(input_channels, hidden_channels, bias=False)
        self.nextus1 = nn.Linear(hidden_channels,vertical_channels, bias=False)
        self.activation = nn.GELU()
        self.nextus2 = nn.Linear(vertical_channels,output_channels, bias=False)


    def l2_norm(self, x, dim=-1):
        norm = torch.norm(x, p=2, dim=dim, keepdim=True)
        norm_outputs = x / (norm + self.eps) * math.sqrt(self.output_channels)
        return norm_outputs

    def forward(self, inputs):
        hidden = self.nextus0(inputs)
        hidden = self.l2_norm(hidden, dim=-1)
        hidden = self.activation(hidden)
        hidden = self.nextus1(hidden)
        hidden = self.l2_norm(hidden, dim=-1)
        hidden = self.activation(hidden)
        output = self.nextus2(hidden)
        return output