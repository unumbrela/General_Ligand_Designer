"""
适配体设计专用工具模块
解决原有代码中的硬编码和兼容性问题，实现RNA/DNA适配体设计功能

作者: AI Assistant
功能: 将BoltzDesign1从"设计蛋白质结合DNA/RNA"改造为"设计RNA/DNA适配体结合蛋白质"
"""

import torch
import numpy as np
import random
import yaml
import sys
import os
from pathlib import Path

# 确保能找到boltz模块
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'boltz', 'src'))

try:
    from boltz.data import const
except ImportError:
    # 如果还是找不到，尝试另一个路径
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'boltz', 'src'))
    from boltz.data import const

class AptamerDesignConfig:
    """适配体设计配置类 - 解决硬编码问题"""
    
    def __init__(self, aptamer_type='RNA', aptamer_chain='A', target_chains=['B'], target_type='protein'):
        self.aptamer_type = aptamer_type.upper()
        self.aptamer_chain = aptamer_chain
        self.target_chains = target_chains
        self.target_type = target_type  # 'protein' 或 'ligand'
        
        # 动态生成允许的token (解决Token ID硬编码问题)
        if self.aptamer_type == 'RNA':
            allowed_tokens_names = ['A', 'G', 'C', 'U', 'N']
            self.nucleotide_alphabet = ['A', 'G', 'C', 'U', 'N']
        elif self.aptamer_type == 'DNA':
            allowed_tokens_names = ['DA', 'DG', 'DC', 'DT', 'DN']
            self.nucleotide_alphabet = ['A', 'G', 'C', 'T', 'N']  # 显示用T
        else:
            raise ValueError(f"Unsupported aptamer type: {aptamer_type}")
        
        # 动态获取token索引 (替代硬编码)
        self.allowed_tokens = [const.token_ids[token] for token in allowed_tokens_names]
        
        # 生成禁止的token列表
        all_tokens = set(range(len(const.tokens)))
        self.forbidden_tokens = list(all_tokens - set(self.allowed_tokens))
        
        # token范围用于序列历史记录 (解决序列历史记录问题)
        self.token_start = min(self.allowed_tokens)
        self.token_end = max(self.allowed_tokens) + 1

def create_aptamer_yaml(target_protein_seq, aptamer_config, name="aptamer_design"):
    """为蛋白质目标创建适配体设计的YAML输入"""
    sequences = []
    
    # 适配体序列 (设计对象 - 角色互换的核心)
    aptamer_entry = {
        aptamer_config.aptamer_type.lower(): {
            "id": [aptamer_config.aptamer_chain],
            "sequence": "N" * 50,  # 占位符，会被随机初始化替换
            "msa": "empty"
        }
    }
    sequences.append(aptamer_entry)
    
    # 目标蛋白质 (固定目标 - 角色互换的核心)
    for i, target_chain in enumerate(aptamer_config.target_chains):
        protein_entry = {
            "protein": {
                "id": [target_chain],
                "sequence": target_protein_seq,
                "msa": "empty"
            }
        }
        sequences.append(protein_entry)
    
    return {"version": 1, "sequences": sequences}

def create_ligand_aptamer_yaml(target_ligand_smiles, aptamer_config, name="ligand_aptamer_design"):
    """为小分子目标创建适配体设计的YAML输入"""
    sequences = []
    
    # 适配体序列 (设计对象 - 角色互换的核心)
    aptamer_entry = {
        aptamer_config.aptamer_type.lower(): {
            "id": [aptamer_config.aptamer_chain],
            "sequence": "N" * 50,  # 占位符，会被随机初始化替换
            "msa": "empty"
        }
    }
    sequences.append(aptamer_entry)
    
    # 目标小分子 (固定目标 - 新增小分子支持)
    for i, target_chain in enumerate(aptamer_config.target_chains):
        ligand_entry = {
            "ligand": {
                "id": [target_chain],
                "smiles": target_ligand_smiles
            }
        }
        sequences.append(ligand_entry)
    
    return {"version": 1, "sequences": sequences}

def save_aptamer_yaml(yaml_content, output_path):
    """保存YAML文件"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    return output_path

def initialize_aptamer_sequence(data, aptamer_config, length):
    """初始化适配体序列 (解决序列初始化问题)"""
    
    if aptamer_config.aptamer_type == 'RNA':
        nucleotides = ['A', 'G', 'C', 'U']
    elif aptamer_config.aptamer_type == 'DNA':
        nucleotides = ['A', 'G', 'C', 'T']
    
    # 生成随机序列
    sequence = ''.join(random.choices(nucleotides, k=length))
    
    # 查找适配体链在sequences中的位置并更新
    for i, seq_entry in enumerate(data['sequences']):
        if aptamer_config.aptamer_type.lower() in seq_entry:
            if seq_entry[aptamer_config.aptamer_type.lower()]["id"][0] == aptamer_config.aptamer_chain:
                data['sequences'][i][aptamer_config.aptamer_type.lower()]['sequence'] = sequence
                print(f"初始化{aptamer_config.aptamer_type}适配体序列 (长度{length}): {sequence}")
                break
    
    return data

def update_aptamer_sequence(opt, batch, mask, aptamer_config, alpha=2.0, device=None):
    """适配体序列更新函数 (替代原update_sequence - 解决核心优化逻辑)"""
    
    batch["logits"] = alpha * batch['res_type_logits']
    
    # 动态创建禁止token掩码 (替代硬编码的token列表)
    forbidden_mask = torch.zeros(batch['logits'].shape[-1], device=device)
    forbidden_mask[aptamer_config.forbidden_tokens] = 1
    
    # 应用掩码
    X = batch['logits'] - forbidden_mask * 1e10
    batch['soft'] = torch.softmax(X/opt["temp"], dim=-1)
    batch['hard'] = torch.zeros_like(batch['soft']).scatter_(-1, batch['soft'].max(dim=-1, keepdim=True)[1], 1.0)
    batch['hard'] = (batch['hard'] - batch['soft']).detach() + batch['soft']
    batch['pseudo'] = opt["soft"] * batch["soft"] + (1-opt["soft"]) * batch["res_type_logits"]
    batch['pseudo'] = opt["hard"] * batch["hard"] + (1-opt["hard"]) * batch["pseudo"]
    batch['res_type'] = batch['pseudo']*mask + batch['res_type_logits']*(1-mask)
    
    # 处理MSA兼容性 (解决MSA处理问题)
    batch['msa'] = batch['res_type'].unsqueeze(0).to(device).detach()
    batch['profile'] = batch['msa'].float().mean(dim=0).to(device).detach()
    
    return batch

def apply_aptamer_gradient_mask(batch, aptamer_config, chain_to_number, device=None):
    """应用适配体梯度掩码 (解决Entity ID和梯度掩码问题)"""
    
    if batch['res_type_logits'].grad is not None:
        # 只对适配体链进行梯度更新 (解决Entity ID问题)
        aptamer_entity_mask = batch['entity_id'] == chain_to_number[aptamer_config.aptamer_chain]
        batch['res_type_logits'].grad[~aptamer_entity_mask, :] = 0
        
        # 动态禁止非适配体token的梯度 (替代硬编码)
        batch['res_type_logits'].grad[..., aptamer_config.forbidden_tokens] = 0

def extract_aptamer_sequence(batch, aptamer_config, chain_to_number):
    """提取设计的适配体序列 (解决字母表映射问题)"""
    
    # 获取适配体链的token
    aptamer_mask = batch['entity_id'] == chain_to_number[aptamer_config.aptamer_chain]
    aptamer_tokens = torch.argmax(batch['res_type'][aptamer_mask, :], dim=-1).detach().cpu().numpy()
    
    # 转换为核酸序列 (使用动态字母表)
    sequence = []
    for token in aptamer_tokens:
        if token in aptamer_config.allowed_tokens:
            # 计算在字母表中的索引
            alphabet_idx = token - aptamer_config.token_start
            if 0 <= alphabet_idx < len(aptamer_config.nucleotide_alphabet):
                sequence.append(aptamer_config.nucleotide_alphabet[alphabet_idx])
    
    return ''.join(sequence)

def record_aptamer_sequence_history(batch, aptamer_config, chain_to_number):
    """记录适配体序列历史 (解决序列历史记录问题)"""
    
    # 获取适配体链的掩码
    aptamer_mask = batch['entity_id'] == chain_to_number[aptamer_config.aptamer_chain]
    
    # 只记录适配体相关的token
    sequence_data = batch['res_type'][0, aptamer_mask, aptamer_config.token_start:aptamer_config.token_end].detach().cpu().numpy()
    
    return sequence_data

def calculate_aptamer_constraints(batch, aptamer_config, chain_to_number, target_type='protein'):
    """计算适配体特异性约束 (添加核酸特异性损失)"""
    
    constraints = {}
    
    # 获取适配体部分的序列概率
    aptamer_mask = batch['entity_id'] == chain_to_number[aptamer_config.aptamer_chain]
    sequence_probs = torch.softmax(batch['res_type_logits'][aptamer_mask, :], dim=-1)
    
    if sequence_probs.numel() == 0:  # 防止空tensor
        return {'gc_content_loss': torch.tensor(0.0, device=batch['res_type_logits'].device)}
    
    # GC含量约束
    if aptamer_config.aptamer_type == 'RNA':
        # G, C 在 RNA 中的绝对索引
        g_idx = const.token_ids['G']  # 24
        c_idx = const.token_ids['C']  # 25
    else:  # DNA
        # DG, DC 在 DNA 中的绝对索引
        g_idx = const.token_ids['DG']  # 29
        c_idx = const.token_ids['DC']  # 30
    
    gc_content = sequence_probs[:, [g_idx, c_idx]].sum(dim=-1).mean()
    
    # 常规的GC含量约束 (符合生物学标准)
    if target_type == 'ligand':
        # 小分子结合：适度提高GC含量，但不过度约束
        gc_target = 0.5   # 50% GC含量 (生物学标准)
        gc_weight = 0.2   # 适中权重 (不压制结构损失)
    else:
        # 蛋白质结合使用标准GC含量
        gc_target = 0.5   # 50% GC含量  
        gc_weight = 0.1   # 标准权重
    
    # 使用平方损失而不是绝对值损失 (更平滑的梯度)
    gc_loss = (gc_content - gc_target) ** 2
    constraints['gc_content_loss'] = gc_loss * gc_weight
    
    # 小分子特异性约束：温和的序列多样性鼓励
    if target_type == 'ligand':
        # 鼓励适度的序列多样性，避免单一核苷酸重复
        if aptamer_config.aptamer_type == 'RNA':
            # RNA: 计算所有核苷酸的分布熵
            nucleotide_indices = [const.token_ids['A'], const.token_ids['G'], 
                                const.token_ids['C'], const.token_ids['U']]
        else:  # DNA
            # DNA: 计算所有核苷酸的分布熵
            nucleotide_indices = [const.token_ids['DA'], const.token_ids['DG'], 
                                const.token_ids['DC'], const.token_ids['DT']]
        
        # 计算序列多样性 (熵约束)
        nucleotide_probs = sequence_probs[:, nucleotide_indices]
        # 避免极端分布，鼓励适度多样性
        diversity_loss = -torch.sum(nucleotide_probs.mean(dim=0) * torch.log(nucleotide_probs.mean(dim=0) + 1e-8))
        # 反向：惩罚低多样性
        constraints['diversity_loss'] = (2.0 - diversity_loss) * 0.05  # 温和的多样性约束
    
    return constraints

def create_aptamer_mask_and_chain_mask(batch, aptamer_config, chain_to_number):
    """创建适配体专用的掩码 (解决掩码生成问题)"""
    
    # 优化掩码：只对适配体链进行优化
    mask = torch.zeros_like(batch['res_type_logits'])
    aptamer_entity_mask = batch['entity_id'] == chain_to_number[aptamer_config.aptamer_chain]
    mask[aptamer_entity_mask, :] = 1
    
    # 链掩码：用于梯度归一化
    chain_mask = aptamer_entity_mask.int()
    
    return mask, chain_mask

def get_aptamer_alphabet(aptamer_type):
    """获取适配体字母表 (解决字母表映射问题)"""
    if aptamer_type.upper() == 'RNA':
        return ['A', 'G', 'C', 'U', 'N']
    elif aptamer_type.upper() == 'DNA':
        return ['A', 'G', 'C', 'T', 'N']
    else:
        raise ValueError(f"Unsupported aptamer type: {aptamer_type}")

def validate_aptamer_design(aptamer_sequence, aptamer_type):
    """验证适配体设计的基本特征"""
    
    metrics = {}
    
    if not aptamer_sequence:
        return {'error': 'Empty sequence'}
    
    # 计算GC含量
    if aptamer_type.upper() == 'RNA':
        gc_count = aptamer_sequence.count('G') + aptamer_sequence.count('C')
    else:  # DNA
        gc_count = aptamer_sequence.count('G') + aptamer_sequence.count('C')
    
    metrics['gc_content'] = gc_count / len(aptamer_sequence) if len(aptamer_sequence) > 0 else 0
    metrics['length'] = len(aptamer_sequence)
    metrics['sequence'] = aptamer_sequence
    
    return metrics

# 导出的主要函数
__all__ = [
    'AptamerDesignConfig',
    'create_aptamer_yaml', 
    'save_aptamer_yaml',
    'initialize_aptamer_sequence',
    'update_aptamer_sequence',
    'apply_aptamer_gradient_mask',
    'extract_aptamer_sequence',
    'record_aptamer_sequence_history',
    'calculate_aptamer_constraints',
    'create_aptamer_mask_and_chain_mask',
    'get_aptamer_alphabet',
    'validate_aptamer_design'
]
