# 🧬 BoltzDesign1 适配体设计功能

## 📋 **功能概述**

BoltzDesign1 现已支持RNA/DNA适配体设计！通过角色互换技术，将原本的"设计蛋白质结合核酸"转换为"设计核酸适配体结合蛋白质"。

## 🎯 **核心特性**

- ✅ **完整角色互换**: 设计RNA/DNA适配体而非蛋白质
- ✅ **保持向后兼容**: 原有蛋白质设计功能完全保留
- ✅ **动态参数配置**: 彻底解决硬编码问题
- ✅ **适配体特异性约束**: GC含量控制、序列验证等
- ✅ **无需重新训练**: 直接使用预训练Boltz模型权重

## 🚀 **快速开始**

### **1. 环境准备**

```bash
# 确保已安装Boltz模型
python -c "from boltz.main import download; from pathlib import Path; download(Path('~/.boltz').expanduser())"

# 验证基础功能
python test_aptamer_design.py
python test_integration.py
```

### **2. RNA适配体设计**

```bash
python boltzdesign.py \
    --design_mode aptamer \
    --aptamer_type RNA \
    --target_protein_seq "MKLLVVVGGVGSGKTTLLRQLAKEFG" \
    --aptamer_length 40 \
    --target_name myprotein \
    --gpu_id 0
```

### **3. DNA适配体设计**

```bash
python boltzdesign.py \
    --design_mode aptamer \
    --aptamer_type DNA \
    --target_protein_seq "MKLLVVVGGVGSGKTTLLRQLAKEFG" \
    --aptamer_length 50 \
    --target_name myprotein \
    --gpu_id 0
```

## 📚 **参数说明**

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--design_mode` | 是 | `protein` | 设计模式：`protein`(原始) 或 `aptamer`(适配体) |
| `--aptamer_type` | 适配体模式下必需 | `RNA` | 适配体类型：`RNA` 或 `DNA` |
| `--target_protein_seq` | 适配体模式下必需 | - | 目标蛋白质序列 |
| `--aptamer_length` | 否 | `40` | 适配体长度（推荐20-80） |
| `--aptamer_chain` | 否 | `A` | 适配体链ID |
| `--target_protein_chains` | 否 | `B` | 目标蛋白质链ID |
| `--target_name` | 否 | `target` | 项目名称 |
| `--gpu_id` | 否 | `0` | 使用的GPU ID |

## 🔧 **技术实现**

### **角色互换核心逻辑**

**原始模式 (蛋白质设计)**:
```yaml
sequences:
- protein: { sequence: "XXX" }  # 设计对象 (变化)
- dna: { sequence: "ATGC" }     # 固定目标
```

**适配体模式 (角色互换)**:
```yaml
sequences:
- rna: { sequence: "NNN" }      # 设计对象 (变化) ← 互换
- protein: { sequence: "ACDEF" } # 固定目标 ← 互换
```

### **解决的关键问题**

1. **Token ID硬编码** → 动态token查找
2. **序列历史记录** → 适配核酸token范围
3. **字母表映射** → 多类型字母表支持
4. **MSA兼容性** → 单序列模式优化
5. **梯度掩码** → 动态entity检测

## 📁 **新增文件**

```
boltzdesign/
├── aptamer_design_utils.py           # 适配体设计核心模块
├── configs/
│   ├── aptamer_rna_config.yaml      # RNA适配体配置
│   └── aptamer_dna_config.yaml      # DNA适配体配置
├── boltzdesign_utils.py             # 修改：添加适配体支持
└── boltzdesign.py                   # 修改：适配体命令行接口

test_aptamer_design.py               # 功能测试脚本
test_integration.py                  # 集成测试脚本
APTAMER_DESIGN_README.md            # 本说明文档
```

## 🧪 **测试验证**

### **基础功能测试**
```bash
python test_aptamer_design.py
# 预期输出: 🎉 所有测试通过! 适配体设计功能就绪!
```

### **集成测试**
```bash
python test_integration.py
# 预期输出: 🎉 所有集成测试通过! 适配体设计基础功能就绪!
```

### **命令行测试**
```bash
# 测试参数解析
python boltzdesign.py --design_mode aptamer --help

# 测试错误处理
python boltzdesign.py --design_mode aptamer --aptamer_type RNA --target_protein_seq "" --aptamer_length 30 --target_name test
# 预期输出: ❌ 错误: 适配体设计模式需要目标蛋白质序列
```

## ⚠️ **注意事项**

1. **模型权重**: 确保已下载Boltz模型权重文件到 `~/.boltz/boltz1_conf.ckpt`
2. **GPU内存**: 适配体设计需要足够的GPU内存，建议使用8GB+显卡
3. **序列长度**: 适配体长度建议控制在20-80之间以获得最佳结果
4. **蛋白质序列**: 目标蛋白质序列应为有效的氨基酸序列

## 🔍 **故障排除**

### **常见问题**

**问题**: `ModuleNotFoundError: No module named 'boltz.data'`
**解决**: 确保项目根目录下有正确的boltz/src路径，并且Python路径设置正确

**问题**: `找不到Boltz模型权重文件`
**解决**: 运行模型下载命令：
```bash
python -c "from boltz.main import download; from pathlib import Path; download(Path('~/.boltz').expanduser())"
```

**问题**: `CUDA out of memory`
**解决**: 减少适配体长度或使用更少的batch size

## 📈 **性能优化建议**

1. **适配体长度**: 从较短长度(20-30)开始测试
2. **迭代次数**: 可以根据需要调整配置文件中的迭代参数
3. **GPU使用**: 确保使用最新的CUDA驱动和PyTorch版本
4. **内存管理**: 设计完成后及时清理GPU内存

## 🎯 **预期结果**

成功运行后，您将获得：
- 设计的适配体序列（RNA或DNA）
- 序列质量验证结果（GC含量、长度等）
- 结构预测结果（如果配置了后续分析步骤）

