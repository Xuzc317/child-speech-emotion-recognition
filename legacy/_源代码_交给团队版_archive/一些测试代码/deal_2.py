from collections import defaultdict
import random

# 修改输入输出路径
in_file = r'D:\大学\毕设\IS2009EmotionChallenge\IS2009EmotionChallenge\labels\IS2009EmotionChallenge\chunk_labels_5cl_corpus.txt'
out_file = r'D:\大学\毕设\IS2009EmotionChallenge\IS2009EmotionChallenge\labels\word_labels_5cl_corpus_out_max.txt'

with open(in_file, 'r') as f_in, open(out_file, 'w') as f_out:
    label_counter = defaultdict(list)
    
    # 第一遍读取：收集所有有效数据
    for line_num, line in enumerate(f_in, 1):
        parts = line.strip().split()
        
        # 基础校验：至少包含文件名、标签和数值
        if len(parts) < 3:
            print(f"跳过不完整行（第{line_num}行）: {line.strip()}")
            continue
        
        filename = parts[0]
        label = parts[1]  # 直接取第二个元素作为标签
        
        # 验证文件名格式（包含3个下划线）
        if filename.count('_') != 3:
            print(f"跳过格式错误文件名（第{line_num}行）: {filename}")
            continue
        
        # 验证标签有效性
        if label not in ['N', 'E', 'R', 'A', 'P']:
            print(f"跳过无效标签（第{line_num}行）: {label}")
            continue
        
        label_counter[label].append(filename)

    # 对N和E进行采样
    for label in ['N', 'E']:
        if label in label_counter and len(label_counter[label]) > 3600:
            label_counter[label] = random.sample(label_counter[label], 3600)

    # 写入最终文件（格式：标签 完整文件名）
    for label in ['N', 'E', 'R', 'A', 'P']:
        for filename in label_counter.get(label, []):
            f_out.write(f"{label} {filename}\n")