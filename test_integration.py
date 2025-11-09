#!/usr/bin/env python3
"""
集成测试脚本 - 测试适配体设计的完整调用链
"""

import os
import sys
import traceback

# 添加路径
sys.path.append(f'{os.getcwd()}/boltzdesign')
sys.path.append(f'{os.getcwd()}/boltz/src')

def test_aptamer_yaml_generation():
    """测试适配体YAML生成和保存"""
    print("🧪 测试: 适配体YAML生成")
    
    try:
        from aptamer_design_utils import AptamerDesignConfig, create_aptamer_yaml, save_aptamer_yaml
        
        # 创建RNA适配体配置
        config = AptamerDesignConfig('RNA', 'A', ['B'])
        
        # 创建YAML
        yaml_content = create_aptamer_yaml("MKLLVVV", config, "test_aptamer")
        
        # 保存YAML
        yaml_path = save_aptamer_yaml(yaml_content, "/tmp/test_aptamer.yaml")
        
        print(f"✅ YAML已保存到: {yaml_path}")
        print(f"✅ 内容预览: {len(yaml_content['sequences'])} 个序列")
        
        return True
        
    except Exception as e:
        print(f"❌ YAML生成测试失败: {e}")
        traceback.print_exc()
        return False

def test_aptamer_config_creation():
    """测试适配体配置创建"""
    print("🧪 测试: 适配体配置创建")
    
    try:
        from aptamer_design_utils import AptamerDesignConfig
        
        # 测试RNA配置
        rna_config = AptamerDesignConfig('RNA', 'A', ['B'])
        print(f"✅ RNA配置: 允许token数量 = {len(rna_config.allowed_tokens)}")
        print(f"✅ RNA配置: 禁止token数量 = {len(rna_config.forbidden_tokens)}")
        
        # 测试DNA配置
        dna_config = AptamerDesignConfig('DNA', 'A', ['B'])
        print(f"✅ DNA配置: 允许token数量 = {len(dna_config.allowed_tokens)}")
        print(f"✅ DNA配置: 禁止token数量 = {len(dna_config.forbidden_tokens)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置创建测试失败: {e}")
        traceback.print_exc()
        return False

def test_boltz_import():
    """测试Boltz模块导入"""
    print("🧪 测试: Boltz模块导入")
    
    try:
        from boltz.data import const
        print(f"✅ 成功导入boltz.data.const")
        print(f"✅ Token总数: {len(const.tokens)}")
        print(f"✅ RNA token示例: {[const.tokens[i] for i in [23,24,25,26,27]]}")
        print(f"✅ DNA token示例: {[const.tokens[i] for i in [28,29,30,31,32]]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Boltz导入测试失败: {e}")
        traceback.print_exc()
        return False

def test_command_line_parsing():
    """测试命令行参数解析"""
    print("🧪 测试: 命令行参数解析")
    
    try:
        # 模拟命令行参数
        import argparse
        sys.argv = [
            'boltzdesign.py',
            '--design_mode', 'aptamer',
            '--aptamer_type', 'RNA', 
            '--target_protein_seq', 'MKLLVVV',
            '--aptamer_length', '30',
            '--target_name', 'test'
        ]
        
        # 导入并测试参数解析
        from boltzdesign import parse_arguments
        args = parse_arguments()
        
        print(f"✅ 设计模式: {args.design_mode}")
        print(f"✅ 适配体类型: {args.aptamer_type}")
        print(f"✅ 目标蛋白质序列: {args.target_protein_seq}")
        print(f"✅ 适配体长度: {args.aptamer_length}")
        
        return True
        
    except Exception as e:
        print(f"❌ 命令行解析测试失败: {e}")
        traceback.print_exc()
        return False

def test_key_functions():
    """测试关键功能函数"""
    print("🧪 测试: 关键功能函数")
    
    try:
        from aptamer_design_utils import (
            get_aptamer_alphabet,
            validate_aptamer_design,
            extract_aptamer_sequence
        )
        
        # 测试字母表函数
        rna_alphabet = get_aptamer_alphabet('RNA')
        dna_alphabet = get_aptamer_alphabet('DNA')
        print(f"✅ RNA字母表: {rna_alphabet}")
        print(f"✅ DNA字母表: {dna_alphabet}")
        
        # 测试序列验证
        validation = validate_aptamer_design("AGCUAGCU", 'RNA')
        print(f"✅ 序列验证: 长度={validation['length']}, GC含量={validation['gc_content']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 关键功能测试失败: {e}")
        traceback.print_exc()
        return False

def run_integration_tests():
    """运行集成测试"""
    print("🧬" + "="*60)
    print("🚀 开始适配体设计集成测试")
    print("🧬" + "="*60)
    
    tests = [
        test_boltz_import,
        test_aptamer_config_creation,
        test_aptamer_yaml_generation,
        test_key_functions,
        test_command_line_parsing,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        print(f"\n{'='*50}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_func.__name__} 通过")
            else:
                failed += 1
                print(f"❌ {test_func.__name__} 失败")
        except Exception as e:
            failed += 1
            print(f"❌ {test_func.__name__} 异常: {e}")
    
    print(f"\n{'='*60}")
    print(f"📊 集成测试结果: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("🎉 所有集成测试通过! 适配体设计基础功能就绪!")
        print("\n📋 后续步骤:")
        print("1. 确认Boltz模型权重文件存在")
        print("2. 运行完整的适配体设计测试")
        print("3. 验证生成的适配体序列质量")
    else:
        print(f"⚠️  {failed} 个测试失败，需要修复后再进行完整测试")
    
    return failed == 0

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
