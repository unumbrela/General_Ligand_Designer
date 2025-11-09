"""
适配体设计专用工具模块
严格按照BoltzDesign1的思路实现RNA/DNA适配体设计
基于论文: BoltzDesign1: Inverting All-Atom Structure Prediction Model

核心原理:
1. 角色互换: 适配体(RNA/DNA) = binder, 蛋白质/小分子 = target  
2. 序列空间转换: 20种氨基酸 → 4-5种核苷酸
3. 损失函数调整: 添加核酸特异性约束(GC含量、碱基配对等)
4. 梯度掩码: 只优化适配体链，保持目标固定
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
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'boltz', 'src'))
    from boltz.data import const


class AptamerDesignConfig:
    """
    适配体设计配置类
    动态管理token索引，避免硬编码问题
    """
    
    def __init__(self, aptamer_type='RNA', aptamer_chain='A', target_chains=['B'], target_type='protein'):
        self.aptamer_type = aptamer_type.upper()
        self.aptamer_chain = aptamer_chain
        self.target_chains = target_chains if isinstance(target_chains, list) else [target_chains]
        self.target_type = target_type  # 'protein' 或 'ligand'
        
        # 动态生成允许的token (核心：避免硬编码)
        if self.aptamer_type == 'RNA':
            self.allowed_token_names = ['A', 'G', 'C', 'U', 'N']
            self.nucleotide_alphabet = ['A', 'G', 'C', 'U', 'N']  # 显示用
            self.nucleotide_alphabet_no_n = ['A', 'G', 'C', 'U']  # 用于序列生成
        elif self.aptamer_type == 'DNA':
            self.allowed_token_names = ['DA', 'DG', 'DC', 'DT', 'DN']
            self.nucleotide_alphabet = ['A', 'G', 'C', 'T', 'N']  # 显示用T
            self.nucleotide_alphabet_no_n = ['A', 'G', 'C', 'T']
        else:
            raise ValueError(f"Unsupported aptamer type: {aptamer_type}")
        
        # 动态获取token索引 (替代硬编码) - 论文的核心思想
        try:
            self.allowed_tokens = [const.token_ids[token] for token in self.allowed_token_names]
        except KeyError as e:
            raise RuntimeError(f"Token {e} not found in const.token_ids. Available tokens: {list(const.token_ids.keys())}")
        
        # 生成禁止的token列表 (所有非核酸token)
        all_tokens = set(range(len(const.tokens)))
        self.forbidden_tokens = list(all_tokens - set(self.allowed_tokens))
        
        # Token范围用于序列历史记录
        self.token_start = min(self.allowed_tokens)
        self.token_end = max(self.allowed_tokens) + 1
        self.num_tokens = len(self.allowed_tokens)  # 5 for RNA/DNA (including N)
        
        # GC索引 (用于GC含量计算)
        if self.aptamer_type == 'RNA':
            self.g_idx = const.token_ids['G']
            self.c_idx = const.token_ids['C']
        else:  # DNA
            self.g_idx = const.token_ids['DG']
            self.c_idx = const.token_ids['DC']
        
        print(f"✅ 适配体配置初始化完成:")
        print(f"   类型: {self.aptamer_type}")
        print(f"   允许的tokens: {self.allowed_token_names} → {self.allowed_tokens}")
        print(f"   禁止的tokens数量: {len(self.forbidden_tokens)}")
        print(f"   GC索引: G={self.g_idx}, C={self.c_idx}")


def create_aptamer_yaml(target_protein_seq, aptamer_config, name="aptamer_design"):
    """
    为蛋白质目标创建适配体设计的YAML输入
    角色互换: aptamer=binder (设计对象), protein=target (固定)
    """
    sequences = []
    
    # 1. 适配体序列 (设计对象 - 会被优化)
    aptamer_entry = {
        aptamer_config.aptamer_type.lower(): {
            "id": [aptamer_config.aptamer_chain],
            "sequence": "N" * 50,  # 占位符，会被随机初始化替换
            "msa": "empty"  # 适配体设计不使用MSA
        }
    }
    sequences.append(aptamer_entry)
    
    # 2. 目标蛋白质 (固定目标 - 不会被优化)
    for target_chain in aptamer_config.target_chains:
        protein_entry = {
            "protein": {
                "id": [target_chain],
                "sequence": target_protein_seq,
                "msa": "empty"  # 可选：如果有MSA可以提供
            }
        }
        sequences.append(protein_entry)
    
    return {"version": 1, "sequences": sequences}


def create_ligand_aptamer_yaml(target_ligand_smiles, aptamer_config, name="ligand_aptamer_design"):
    """
    为小分子目标创建适配体设计的YAML输入
    角色互换: aptamer=binder, ligand=target
    """
    sequences = []
    
    # 1. 适配体序列 (设计对象)
    aptamer_entry = {
        aptamer_config.aptamer_type.lower(): {
            "id": [aptamer_config.aptamer_chain],
            "sequence": "N" * 50,
            "msa": "empty"
        }
    }
    sequences.append(aptamer_entry)
    
    # 2. 目标小分子 (固定目标)
    for target_chain in aptamer_config.target_chains:
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
    """
    初始化适配体序列
    遵循BoltzDesign1的思路: 随机初始化后通过梯度优化
    
    Args:
        data: YAML数据字典
        aptamer_config: AptamerDesignConfig对象
        length: 适配体长度
    
    Returns:
        更新后的data字典
    """
    # 生成随机核酸序列 (不包含N)
    sequence = ''.join(random.choices(aptamer_config.nucleotide_alphabet_no_n, k=length))
    
    # 查找适配体链在sequences中的位置并更新
    found = False
    for i, seq_entry in enumerate(data['sequences']):
        if aptamer_config.aptamer_type.lower() in seq_entry:
            if seq_entry[aptamer_config.aptamer_type.lower()]["id"][0] == aptamer_config.aptamer_chain:
                data['sequences'][i][aptamer_config.aptamer_type.lower()]['sequence'] = sequence
                print(f"🧬 初始化{aptamer_config.aptamer_type}适配体序列 (长度{length}): {sequence}")
                found = True
                break
    
    if not found:
        raise ValueError(f"未找到适配体链 {aptamer_config.aptamer_chain} in YAML sequences")
    
    return data


def update_aptamer_sequence(opt, batch, mask, aptamer_config, alpha=2.0, device=None):
    """
    适配体序列更新函数
    严格遵循BoltzDesign1的四阶段优化策略
    
    对应论文的公式:
    - Stage 1 (warm-up): sequence = softmax(logits)
    - Stage 2 (soft): sequence = (1-λ)*logits + λ*softmax(logits)  
    - Stage 3 (temp annealing): sequence = softmax(logits/temp)
    - Stage 4 (hard): sequence = one_hot (with straight-through)
    
    Args:
        opt: 优化参数字典 {'soft', 'hard', 'temp'}
        batch: 批次数据
        mask: 适配体掩码
        aptamer_config: 适配体配置
        alpha: logits缩放因子
        device: 设备
    
    Returns:
        更新后的batch
    """
    # 1. 缩放logits (论文中的alpha参数)
    batch["logits"] = alpha * batch['res_type_logits']
    
    # 2. 创建禁止token掩码 (动态，不硬编码)
    forbidden_mask = torch.zeros(batch['logits'].shape[-1], device=device)
    forbidden_mask[aptamer_config.forbidden_tokens] = 1e10  # 大负数使概率接近0
    
    # 3. 应用掩码 (只保留核酸tokens)
    X = batch['logits'] - forbidden_mask
    
    # 4. 四阶段转换 (论文核心算法)
    batch['soft'] = torch.softmax(X / opt["temp"], dim=-1)
    
    # Hard encoding: one-hot with straight-through estimator
    batch['hard'] = torch.zeros_like(batch['soft']).scatter_(
        -1, batch['soft'].max(dim=-1, keepdim=True)[1], 1.0
    )
    batch['hard'] = (batch['hard'] - batch['soft']).detach() + batch['soft']
    
    # Pseudo sequence: 混合soft和hard
    batch['pseudo'] = opt["soft"] * batch["soft"] + (1 - opt["soft"]) * batch["res_type_logits"]
    batch['pseudo'] = opt["hard"] * batch["hard"] + (1 - opt["hard"]) * batch["pseudo"]
    
    # 5. 应用掩码 (只更新适配体部分)
    batch['res_type'] = batch['pseudo'] * mask + batch['res_type_logits'] * (1 - mask)
    
    # 6. 更新MSA (核酸设计使用单序列模式)
    batch['msa'] = batch['res_type'].unsqueeze(0).to(device).detach()
    batch['profile'] = batch['msa'].float().mean(dim=0).to(device).detach()
    
    return batch


def apply_aptamer_gradient_mask(batch, aptamer_config, chain_to_number, device=None):
    """
    应用适配体梯度掩码
    确保只优化适配体链，目标链保持固定
    
    这是论文中"翻转"的关键实现:
    - 蛋白质设计: 优化protein链，固定target链
    - 适配体设计: 优化RNA/DNA链，固定protein链
    """
    if batch['res_type_logits'].grad is not None:
        # 1. 只对适配体链进行梯度更新
        aptamer_entity_mask = batch['entity_id'] == chain_to_number[aptamer_config.aptamer_chain]
        batch['res_type_logits'].grad[~aptamer_entity_mask, :] = 0
        
        # 2. 禁止非核酸token的梯度 (动态，不硬编码)
        batch['res_type_logits'].grad[..., aptamer_config.forbidden_tokens] = 0


def extract_aptamer_sequence(batch, aptamer_config, chain_to_number):
    """
    提取设计的适配体序列
    将token索引转换为核酸字母
    """
    # 获取适配体链的掩码
    aptamer_mask = batch['entity_id'] == chain_to_number[aptamer_config.aptamer_chain]
    
    if not aptamer_mask.any():
        return ""
    
    # 获取token索引
    aptamer_tokens = torch.argmax(batch['res_type'][aptamer_mask, :], dim=-1).detach().cpu().numpy()
    
    # 转换为核酸序列
    sequence = []
    for token in aptamer_tokens:
        if token in aptamer_config.allowed_tokens:
            # 计算在字母表中的索引
            try:
                token_idx = aptamer_config.allowed_tokens.index(token)
                sequence.append(aptamer_config.nucleotide_alphabet[token_idx])
            except (ValueError, IndexError):
                sequence.append('N')  # 未知核苷酸
        else:
            # 非法token，应该不会出现（梯度已被掩码）
            sequence.append('N')
    
    return ''.join(sequence)


def record_aptamer_sequence_history(batch, aptamer_config, chain_to_number):
    """
    记录适配体序列历史
    只记录核酸相关的token概率 (5维: A,G,C,U/T,N)
    
    修复问题: 原始代码假设20维氨基酸，这里改为5维核苷酸
    """
    # 获取适配体链的掩码
    aptamer_mask = batch['entity_id'] == chain_to_number[aptamer_config.aptamer_chain]
    
    if not aptamer_mask.any():
        return np.array([])
    
    # 只记录核酸token的概率 (5维)
    sequence_probs = batch['res_type'][0, aptamer_mask, :]
    nucleotide_probs = sequence_probs[:, aptamer_config.allowed_tokens].detach().cpu().numpy()
    
    return nucleotide_probs


def calculate_aptamer_constraints(batch, aptamer_config, chain_to_number, target_type='protein'):
    """
    计算适配体特异性约束
    论文中提到可以添加自定义损失函数，这里实现核酸特异性约束
    
    约束包括:
    1. GC含量约束 (生物学标准: 40-60%)
    2. 序列多样性约束 (避免poly-A/poly-G)
    3. 碱基配对潜力 (鼓励二级结构)
    """
    constraints = {}
    device = batch['res_type_logits'].device
    
    # 获取适配体部分的序列概率
    aptamer_mask = batch['entity_id'] == chain_to_number[aptamer_config.aptamer_chain]
    
    if not aptamer_mask.any():
        return {'gc_content_loss': torch.tensor(0.0, device=device)}
    
    sequence_probs = torch.softmax(batch['res_type_logits'][aptamer_mask, :], dim=-1)
    
    # ===== 1. GC含量约束 =====
    gc_content = sequence_probs[:, [aptamer_config.g_idx, aptamer_config.c_idx]].sum(dim=-1).mean()
    
    # 生物学上合理的GC含量: 40-60%
    if target_type == 'ligand':
        gc_target = 0.5   # 小分子结合: 50% GC
        gc_weight = 0.15
    else:
        gc_target = 0.5   # 蛋白质结合: 50% GC
        gc_weight = 0.1
    
    # 使用平方损失 (论文中的标准损失形式)
    gc_loss = ((gc_content - gc_target) ** 2) * gc_weight
    constraints['gc_content_loss'] = gc_loss
    
    # ===== 2. 序列多样性约束 (避免单核苷酸重复) =====
    # 计算核苷酸分布的熵
    nucleotide_indices = aptamer_config.allowed_tokens[:4]  # 排除N
    nucleotide_probs = sequence_probs[:, nucleotide_indices]
    
    # 计算全局核苷酸分布
    global_dist = nucleotide_probs.mean(dim=0)
    
    # 熵: H = -Σ p*log(p)
    entropy = -torch.sum(global_dist * torch.log(global_dist + 1e-8))
    
    # 最大熵 = log(4) ≈ 1.386 (均匀分布)
    # 鼓励高熵 (多样性)
    max_entropy = torch.log(torch.tensor(4.0, device=device))
    diversity_loss = (max_entropy - entropy) * 0.05  # 温和的约束
    constraints['diversity_loss'] = diversity_loss
    
    # ===== 3. 碱基配对潜力 (核酸特异性) =====
    # RNA/DNA可以形成二级结构 (stem-loop)
    # 简化版: 鼓励A-U/T和G-C配对的潜力
    if aptamer_config.aptamer_type == 'RNA':
        a_idx_local = aptamer_config.allowed_tokens.index(const.token_ids['A'])
        u_idx_local = aptamer_config.allowed_tokens.index(const.token_ids['U'])
        g_idx_local = aptamer_config.allowed_tokens.index(const.token_ids['G'])
        c_idx_local = aptamer_config.allowed_tokens.index(const.token_ids['C'])
    else:  # DNA
        a_idx_local = aptamer_config.allowed_tokens.index(const.token_ids['DA'])
        u_idx_local = aptamer_config.allowed_tokens.index(const.token_ids['DT'])  # DNA用T
        g_idx_local = aptamer_config.allowed_tokens.index(const.token_ids['DG'])
        c_idx_local = aptamer_config.allowed_tokens.index(const.token_ids['DC'])
    
    # 计算A/U(T)和G/C的平衡性
    a_prob = nucleotide_probs[:, a_idx_local].mean()
    u_prob = nucleotide_probs[:, u_idx_local].mean()
    g_prob = nucleotide_probs[:, g_idx_local].mean()
    c_prob = nucleotide_probs[:, c_idx_local].mean()
    
    # 鼓励配对平衡 (A≈U, G≈C)
    pairing_balance = ((a_prob - u_prob) ** 2 + (g_prob - c_prob) ** 2) * 0.02
    constraints['pairing_balance_loss'] = pairing_balance
    
    return constraints


def create_aptamer_mask_and_chain_mask(batch, aptamer_config, chain_to_number):
    """
    创建适配体专用的掩码
    优化掩码: 只对适配体链进行优化
    链掩码: 用于梯度归一化和损失计算
    """
    # 优化掩码: 只对适配体链进行优化
    mask = torch.zeros_like(batch['res_type_logits'])
    aptamer_entity_mask = batch['entity_id'] == chain_to_number[aptamer_config.aptamer_chain]
    mask[aptamer_entity_mask, :] = 1
    
    # 链掩码: 用于梯度归一化
    chain_mask = aptamer_entity_mask.int()
    
    return mask, chain_mask


def get_aptamer_alphabet(aptamer_type):
    """获取适配体字母表 (用于显示)"""
    if aptamer_type.upper() == 'RNA':
        return ['A', 'G', 'C', 'U', 'N']
    elif aptamer_type.upper() == 'DNA':
        return ['A', 'G', 'C', 'T', 'N']
    else:
        raise ValueError(f"Unsupported aptamer type: {aptamer_type}")


def validate_aptamer_design(aptamer_sequence, aptamer_type):
    """
    验证适配体设计的基本特征
    参考论文中的评估指标
    """
    metrics = {}
    
    if not aptamer_sequence:
        return {'error': 'Empty sequence'}
    
    # 过滤掉未知核苷酸N
    valid_sequence = aptamer_sequence.replace('N', '')
    
    if not valid_sequence:
        return {'error': 'No valid nucleotides'}
    
    # 计算GC含量
    gc_count = valid_sequence.count('G') + valid_sequence.count('C')
    metrics['gc_content'] = gc_count / len(valid_sequence)
    metrics['length'] = len(aptamer_sequence)
    metrics['valid_length'] = len(valid_sequence)
    metrics['sequence'] = aptamer_sequence
    
    # 计算序列复杂度 (熵)
    nucleotide_counts = {nt: valid_sequence.count(nt) for nt in set(valid_sequence)}
    total = len(valid_sequence)
    entropy = -sum((count/total) * np.log2(count/total) for count in nucleotide_counts.values())
    metrics['entropy'] = entropy
    metrics['max_entropy'] = np.log2(4)  # 对于4种核苷酸
    
    # 检测同聚物 (poly-X)
    max_repeat = max(
        max((len(list(g)) for k, g in __import__('itertools').groupby(valid_sequence) if k == nt), default=0)
        for nt in 'AGCUT'
    )
    metrics['max_repeat'] = max_repeat
    
    # 质量评估
    quality = 'Good'
    if metrics['gc_content'] < 0.3 or metrics['gc_content'] > 0.7:
        quality = 'Warning: GC content out of range'
    if max_repeat > 5:
        quality = 'Warning: Long homopolymer detected'
    if entropy / np.log2(4) < 0.7:
        quality = 'Warning: Low sequence complexity'
    
    metrics['quality'] = quality
    
    return metrics


# 导出的主要函数
__all__ = [
    'AptamerDesignConfig',
    'create_aptamer_yaml',
    'create_ligand_aptamer_yaml',
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