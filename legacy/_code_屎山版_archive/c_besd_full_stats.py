import pandas as pd
import matplotlib.pyplot as plt

# ==== 配置 ====
INPUT_CSV = "c_besd_statistics.csv"   # 你的原始统计结果
OUTPUT_SUMMARY = "c_besd_summary.csv" # 汇总后的表格

# ==== 读取数据 ====
df = pd.read_csv(INPUT_CSV)

# ==== 按情绪类别汇总（均值 ± 标准差）====
summary = df.groupby("情感").agg(
    Duration_mean=("时长(s)", "mean"),
    Duration_std=("时长(s)", "std"),
    Silence_mean=("静音比例", "mean"),
    Silence_std=("静音比例", "std"),
    F0_mean=("F0均值(Hz)", "mean"),
    F0_std=("F0均值(Hz)", "std"),
    F0_var=("F0标准差(Hz)", "mean"),
    SNR_mean=("SNR(dB)", "mean"),
    SNR_std=("SNR(dB)", "std"),
    Smooth_mean=("频谱平滑度", "mean"),
    Smooth_std=("频谱平滑度", "std"),
    Syll_mean=("说话速率(syll/sec)", "mean"),
    Syll_std=("说话速率(syll/sec)", "std")
).reset_index()

# ==== 保存汇总结果 ====
summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")
print(f"汇总结果已保存到 {OUTPUT_SUMMARY}")

# ==== 图表 1: 情感类别样本数 ====
plt.figure(figsize=(8, 6))
df["情感"].value_counts().plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("Emotion Category Distribution", fontsize=14)
plt.xlabel("Emotion")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("emotion_distribution.png", dpi=300)
plt.close()

# ==== 图表 2: 时长分布 ====
plt.figure(figsize=(8, 6))
df["时长(s)"].hist(bins=20, color="lightgreen", edgecolor="black")
plt.title("Speech Duration Distribution", fontsize=14)
plt.xlabel("Duration (s)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("duration_distribution.png", dpi=300)
plt.close()

print("统计完成，已输出图表：emotion_distribution.png 和 duration_distribution.png")
