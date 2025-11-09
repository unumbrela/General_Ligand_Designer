#!/usr/bin/env python3
"""
适配体设计测试脚本
测试RNA/DNA适配体设计功能的完整流程
"""

import os
import sys
import yaml
import tempfile
from pathlib import Path

# 添加路径
sys.path.append(f'{os.getcwd()}/boltzdesign')
sys.path.append(f'{os.getcwd()}/boltz/src')

def test_aptamer_config():
    """测试适配体配置创建"""
    print("🧪 测试1: 适配体配置创建")
    
    from aptamer_design_utils import AptamerDesignConfig
    
    # 测试RNA配置
    rna_config = AptamerDesignConfig('RNA', 'A', ['B'])
    assert rna_config.aptamer_type == 'RNA'
    assert rna_config.allowed_tokens == [23, 24, 25, 26, 27]  # A,G,C,U,N
    assert len(rna_config.forbidden_tokens) > 20  # 应该禁止大部分token
    print("  ✅ RNA配置创建成功")
    
    # 测试DNA配置
    dna_config = AptamerDesignConfig('DNA', 'A', ['B'])
    assert dna_config.aptamer_type == 'DNA'
    assert dna_config.allowed_tokens == [28, 29, 30, 31, 32]  # DA,DG,DC,DT,DN
    print("  ✅ DNA配置创建成功")
    
def test_yaml_creation():
    """测试YAML文件创建"""
    print("🧪 测试2: YAML文件创建")
    
    from aptamer_design_utils import AptamerDesignConfig, create_aptamer_yaml, save_aptamer_yaml
    
    # 测试蛋白质序列
    test_protein_seq = "MKLLVVVGGVGSGKTTLLRQLAKEFG"
    
    # 测试RNA YAML
    rna_config = AptamerDesignConfig('RNA', 'A', ['B'])
    rna_yaml = create_aptamer_yaml(test_protein_seq, rna_config)
    
    assert rna_yaml['version'] == 1
    assert len(rna_yaml['sequences']) == 2  # 适配体 + 蛋白质目标
    
    # 检查适配体条目
    aptamer_entry = rna_yaml['sequences'][0]
    assert 'rna' in aptamer_entry
    assert aptamer_entry['rna']['id'] == ['A']
    
    # 检查蛋白质条目
    protein_entry = rna_yaml['sequences'][1]
    assert 'protein' in protein_entry
    assert protein_entry['protein']['sequence'] == test_protein_seq
    
    print("  ✅ RNA YAML创建成功")
    
    # 测试文件保存
    with tempfile.TemporaryDirectory() as temp_dir:
        yaml_path = save_aptamer_yaml(rna_yaml, f"{temp_dir}/test.yaml")
        assert os.path.exists(yaml_path)
        
        # 验证文件内容
        with open(yaml_path, 'r') as f:
            loaded_yaml = yaml.safe_load(f)
        assert loaded_yaml == rna_yaml
        
    print("  ✅ YAML文件保存成功")

def test_sequence_processing():
    """测试序列处理功能"""
    print("🧪 测试3: 序列处理功能")
    
    from aptamer_design_utils import (
        AptamerDesignConfig, get_aptamer_alphabet, 
        validate_aptamer_design, initialize_aptamer_sequence
    )
    
    # 测试字母表
    rna_alphabet = get_aptamer_alphabet('RNA')
    assert rna_alphabet == ['A', 'G', 'C', 'U', 'N']
    
    dna_alphabet = get_aptamer_alphabet('DNA')
    assert dna_alphabet == ['A', 'G', 'C', 'T', 'N']
    
    print("  ✅ 字母表测试成功")
    
    # 测试序列验证
    test_rna_seq = "AGCUAGCUAGCU"
    validation = validate_aptamer_design(test_rna_seq, 'RNA')
    assert validation['length'] == 12
    assert 0 <= validation['gc_content'] <= 1
    
    print("  ✅ 序列验证成功")
    
    # 测试序列初始化
    test_data = {
        'sequences': [{
            'rna': {
                'id': ['A'],
                'sequence': 'N' * 20,
                'msa': 'empty'
            }
        }]
    }
    
    config = AptamerDesignConfig('RNA', 'A', ['B'])
    updated_data = initialize_aptamer_sequence(test_data, config, 20)
    
    # 检查是否生成了随机序列
    new_seq = updated_data['sequences'][0]['rna']['sequence']
    assert len(new_seq) == 20
    assert all(nt in 'AGCU' for nt in new_seq)  # 应该只包含RNA核苷酸
    
    print("  ✅ 序列初始化成功")

def test_token_mask_generation():
    """测试token掩码生成"""
    print("🧪 测试4: Token掩码生成")
    
    from aptamer_design_utils import AptamerDesignConfig
    from boltz.data import const
    
    # 测试RNA token掩码
    rna_config = AptamerDesignConfig('RNA', 'A', ['B'])
    
    # 验证允许的token
    rna_tokens = ['A', 'G', 'C', 'U', 'N']
    expected_allowed = [const.token_ids[token] for token in rna_tokens]
    assert rna_config.allowed_tokens == expected_allowed
    
    # 验证禁止的token不包含允许的
    for allowed in rna_config.allowed_tokens:
        assert allowed not in rna_config.forbidden_tokens
    
    # 验证禁止的token包含蛋白质token
    protein_tokens = ['ALA', 'GLY', 'VAL']  # 示例
    for prot_token in protein_tokens:
        if prot_token in const.token_ids:
            assert const.token_ids[prot_token] in rna_config.forbidden_tokens
    
    print("  ✅ Token掩码生成成功")

def test_complete_workflow():
    """测试完整工作流程"""
    print("🧪 测试5: 完整工作流程")
    
    try:
        from aptamer_design_utils import (
            AptamerDesignConfig, create_aptamer_yaml, save_aptamer_yaml,
            initialize_aptamer_sequence, validate_aptamer_design
        )
        
        # 1. 创建配置
        config = AptamerDesignConfig('RNA', 'A', ['B'])
        
        # 2. 创建YAML
        protein_seq = "MKLLVVV"
        yaml_content = create_aptamer_yaml(protein_seq, config)
        
        # 3. 初始化适配体序列
        yaml_content = initialize_aptamer_sequence(yaml_content, config, 30)
        
        # 4. 保存文件
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_path = save_aptamer_yaml(yaml_content, f"{temp_dir}/workflow_test.yaml")
            
            # 5. 验证文件
            with open(yaml_path, 'r') as f:
                final_yaml = yaml.safe_load(f)
            
            # 6. 提取并验证序列
            aptamer_seq = final_yaml['sequences'][0]['rna']['sequence']
            validation = validate_aptamer_design(aptamer_seq, 'RNA')
            
            assert validation['length'] == 30
            assert 0 <= validation['gc_content'] <= 1
            
        print("  ✅ 完整工作流程测试成功")
        
    except Exception as e:
        print(f"  ❌ 完整工作流程测试失败: {e}")
        raise

def run_all_tests():
    """运行所有测试"""
    print("🧬" + "="*60)
    print("🚀 开始适配体设计功能测试")
    print("🧬" + "="*60)
    
    tests = [
        test_aptamer_config,
        test_yaml_creation,
        test_sequence_processing,
        test_token_mask_generation,
        test_complete_workflow
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} 失败: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"📊 测试结果: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("🎉 所有测试通过! 适配体设计功能就绪!")
    else:
        print("⚠️  部分测试失败，请检查错误信息")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
