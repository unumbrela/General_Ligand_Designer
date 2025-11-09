# comprehensive_evaluation.py
import json
import re
from collections import Counter
import math

def comprehensive_evaluation(sequence, confidence_file_path, aptamer_type='RNA'):
    """
    综合评估适配体设计质量
    参考BoltzDesign1论文的多指标评估体系
    """
    print(f"\n{'='*70}")
    print(f"🧬 RNA/DNA适配体设计质量综合评估")
    print(f"{'='*70}\n")
    
    # ===== 1. 序列质量指标 =====
    print(f"📋 1. 序列质量指标")
    print(f"   序列: {sequence}")
    print(f"   长度: {len(sequence)}")
    
    # N含量
    n_count = sequence.count('N')
    n_score = 100 if n_count == 0 else max(0, 100 - n_count/len(sequence)*200)
    print(f"   ✓ N含量: {n_count} ({n_count/len(sequence)*100:.1f}%) - 得分: {n_score:.0f}/100")
    
    # GC含量
    valid_seq = sequence.replace('N', '')
    gc_count = valid_seq.count('G') + valid_seq.count('C')
    gc_content = gc_count / len(valid_seq) if valid_seq else 0
    gc_score = max(0, 100 - abs(gc_content - 0.5) * 200)  # 50%最优
    print(f"   ✓ GC含量: {gc_content*100:.1f}% - 得分: {gc_score:.0f}/100")
    
    # poly-X检测
    max_poly = 0
    poly_info = []
    for nt in 'AGCU':
        matches = re.findall(f'{nt}{{3,}}', sequence)
        if matches:
            max_len = max(len(m) for m in matches)
            max_poly = max(max_poly, max_len)
            poly_info.append(f"{nt}x{max_len}")
    poly_score = max(0, 100 - max_poly * 20)
    print(f"   ✓ 同聚物: 最长{max_poly} ({', '.join(poly_info) if poly_info else '无'}) - 得分: {poly_score:.0f}/100")
    
    # 序列复杂度
    counts = Counter(valid_seq)
    total = len(valid_seq)
    entropy = -sum((c/total)*math.log2(c/total) for c in counts.values() if c > 0)
    max_entropy = math.log2(4)
    complexity_score = (entropy / max_entropy) * 100
    print(f"   ✓ 序列复杂度: {entropy:.3f}/{max_entropy:.3f} - 得分: {complexity_score:.0f}/100")
    
    sequence_score = (n_score + gc_score + poly_score + complexity_score) / 4
    print(f"\n   📊 序列质量总分: {sequence_score:.1f}/100\n")
    
    # ===== 2. 结构置信度指标 =====
    print(f"🏗️ 2. 结构置信度指标")
    
    try:
        with open(confidence_file_path, 'r') as f:
            conf_data = json.load(f)
        
        # pLDDT
        plddt = conf_data.get('complex_plddt', 0)
        plddt_score = plddt * 100
        plddt_grade = "优秀" if plddt > 0.7 else "中等" if plddt > 0.5 else "低"
        print(f"   ✓ pLDDT: {plddt:.3f} ({plddt_grade}) - 得分: {plddt_score:.0f}/100")
        
        # iPTM (链间接触质量 - 关键!)
        iptm = conf_data.get('iptm', 0)
        iptm_score = max(0, min(100, (iptm - 0.4) / 0.3 * 100))  # 0.4-0.7映射到0-100
        iptm_grade = "优秀" if iptm > 0.6 else "中等" if iptm > 0.4 else "低"
        print(f"   ✓ iPTM: {iptm:.3f} ({iptm_grade}) - 得分: {iptm_score:.0f}/100")
        
        # pTM
        ptm = conf_data.get('ptm', 0)
        ptm_score = max(0, min(100, (ptm - 0.4) / 0.3 * 100))
        print(f"   ✓ pTM: {ptm:.3f} - 得分: {ptm_score:.0f}/100")
        
        # 链间PAE (如果有)
        if 'pair_chains_iptm' in conf_data:
            print(f"   ✓ 链间接触置信度:")
            for chain1, chain2_dict in conf_data['pair_chains_iptm'].items():
                for chain2, value in chain2_dict.items():
                    if chain1 != chain2:
                        print(f"      {chain1}-{chain2}: {value:.3f}")
        
        structure_score = (plddt_score * 0.3 + iptm_score * 0.4 + ptm_score * 0.3)
        print(f"\n   📊 结构质量总分: {structure_score:.1f}/100\n")
        
    except Exception as e:
        print(f"   ⚠️  无法读取置信度文件: {e}")
        structure_score = 0
    
    # ===== 3. 综合评分 =====
    print(f"🎯 3. 综合评分")
    # 序列40%，结构60%（结构更重要）
    final_score = sequence_score * 0.4 + structure_score * 0.6
    
    print(f"   序列质量: {sequence_score:.1f}/100 (权重40%)")
    print(f"   结构质量: {structure_score:.1f}/100 (权重60%)")
    print(f"\n   {'🏆 最终得分:':<15} {final_score:.1f}/100")
    
    if final_score >= 70:
        grade = "✅ 优秀 - 可用于实验验证"
    elif final_score >= 50:
        grade = "⚠️  中等 - 需要进一步优化"
    else:
        grade = "❌ 不合格 - 需要重新设计"
    
    print(f"   {'评级:':<15} {grade}\n")
    
    # ===== 4. 改进建议 =====
    print(f"💡 4. 改进建议")
    suggestions = []
    
    if n_count > 0:
        suggestions.append("   • 序列中含有未确定核苷酸(N)，需要继续优化")
    if abs(gc_content - 0.5) > 0.1:
        suggestions.append(f"   • GC含量({gc_content*100:.1f}%)偏离最优值50%")
    if max_poly > 3:
        suggestions.append(f"   • 存在过长同聚物({max_poly}个连续)，增加poly_penalty权重")
    if plddt < 0.6:
        suggestions.append(f"   • pLDDT过低({plddt:.2f})，建议:")
        suggestions.append("     - 设置 distogram_only: false")
        suggestions.append("     - 增加 recycling_steps: 1-2")
        suggestions.append("     - 增加优化迭代次数")
    if iptm < 0.5:
        suggestions.append(f"   • iPTM过低({iptm:.2f})，适配体-蛋白结合弱，建议:")
        suggestions.append("     - 增加 inter_contact 损失权重")
        suggestions.append("     - 减小 inter_chain_cutoff 距离")
    
    if not suggestions:
        suggestions.append("   ✅ 设计质量良好，无明显改进建议")
    
    for suggestion in suggestions:
        print(suggestion)
    
    print(f"\n{'='*70}\n")
    
    return {
        'sequence_score': sequence_score,
        'structure_score': structure_score,
        'final_score': final_score,
        'plddt': plddt,
        'iptm': iptm,
        'gc_content': gc_content,
        'max_poly': max_poly
    }

# 使用示例
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("用法: python comprehensive_evaluation.py <序列> <confidence.json路径>")
        sys.exit(1)
    
    sequence = sys.argv[1]
    conf_file = sys.argv[2]
    
    comprehensive_evaluation(sequence, conf_file)