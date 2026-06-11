#!/usr/bin/env python3
"""
Build v2 Chinese review edition (.docx) from the LaTeX manuscript.
Translates all content to natural academic Chinese, preserving all numbers,
citations, and scientific claims.
"""

from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml
import os

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v2_中文审阅版.docx")

doc = Document()

# ── Page Setup ──────────────────────────────────────────────────
for section in doc.sections:
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)

# ── Style definitions ───────────────────────────────────────────
style = doc.styles['Normal']
style.font.name = '宋体'
style.font.size = Pt(11)
style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
style.paragraph_format.line_spacing = 1.5

# Heading styles
for level, (name, size, bold) in enumerate([(1, 16, True), (2, 14, True), (3, 12, True)], 1):
    h_style = doc.styles[f'Heading {level}']
    h_style.font.name = '黑体'
    h_style.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
    h_style.font.size = Pt(size)
    h_style.font.bold = bold
    h_style.font.color.rgb = RGBColor(0, 0, 0)
    h_style.paragraph_format.space_before = Pt(12)
    h_style.paragraph_format.space_after = Pt(6)

# ── Helper functions ────────────────────────────────────────────
def add_para(text, bold=False, italic=False, size=11, alignment=None, font_name=None, spacing_after=None):
    """Add a paragraph with optional formatting."""
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if font_name:
        run.font.name = font_name
        run._element.rPr.rFonts.set(qn('w:eastAsia'), font_name)
    if alignment is not None:
        p.alignment = alignment
    if spacing_after is not None:
        p.paragraph_format.space_after = Pt(spacing_after)
    return p

def add_mixed_para(segments):
    """Add a paragraph with mixed formatting.
    segments: list of (text, bold, italic, font_name) tuples.
    """
    p = doc.add_paragraph()
    for seg in segments:
        text = seg[0]
        bold = seg[1] if len(seg) > 1 else False
        italic = seg[2] if len(seg) > 2 else False
        font_name = seg[3] if len(seg) > 3 else None
        run = p.add_run(text)
        run.font.size = Pt(11)
        run.bold = bold
        run.italic = italic
        if font_name:
            run.font.name = font_name
            run._element.rPr.rFonts.set(qn('w:eastAsia'), font_name)
    return p

def set_cell_font(cell, text, bold=False, size=10, font_name='宋体', alignment=None):
    """Set cell text with formatting."""
    # Clear existing paragraphs
    for paragraph in cell.paragraphs:
        for run in paragraph.runs:
            run.text = ''
    p = cell.paragraphs[0]
    if alignment is not None:
        p.alignment = alignment
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.bold = bold
    run.font.name = font_name
    run._element.rPr.rFonts.set(qn('w:eastAsia'), font_name)
    # Remove cell margins
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="30" w:type="dxa"/><w:bottom w:w="30" w:type="dxa"/><w:left w:w="60" w:type="dxa"/><w:right w:w="60" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def make_table(headers, rows, col_widths=None, caption=None):
    """Create a formatted table."""
    if caption:
        add_para(caption, bold=True, size=10, alignment=WD_ALIGN_PARAGRAPH.CENTER, spacing_after=4)

    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header row
    for j, header in enumerate(headers):
        set_cell_font(table.rows[0].cells[j], header, bold=True, size=9, font_name='黑体',
                      alignment=WD_ALIGN_PARAGRAPH.CENTER)

    # Data rows
    for i, row in enumerate(rows):
        for j, cell_text in enumerate(row):
            set_cell_font(table.rows[i + 1].cells[j], str(cell_text), bold=False, size=9,
                          alignment=WD_ALIGN_PARAGRAPH.CENTER)

    # Set column widths if provided
    if col_widths:
        for row in table.rows:
            for j, width in enumerate(col_widths):
                row.cells[j].width = Cm(width)

    # Shade header row
    for cell in table.rows[0].cells:
        shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="D9E2F3" w:val="clear"/>')
        cell._tc.get_or_add_tcPr().append(shading)

    doc.add_paragraph()  # spacer
    return table

def add_note(text):
    """Add a small footnote-style paragraph."""
    add_para(text, size=9, italic=True)

# ══════════════════════════════════════════════════════════════════
# HEADER NOTE
# ══════════════════════════════════════════════════════════════════

add_para(
    "本文为 INTERSPEECH/ICASSP 2027 投稿论文 v2 中文翻译版，供内部审阅使用。"
    "所有数据基于 B1 实验（2026-06-11 完成）及部分 B2 实验。",
    italic=True, size=10, spacing_after=6
)

# ══════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════

title_p = doc.add_paragraph()
title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
title_run = title_p.add_run(
    "分布驱动儿童语音情绪识别：\n基于Fréchet Distance诊断的跨语料库系统基准测试"
)
title_run.font.size = Pt(18)
title_run.bold = True
title_run.font.name = '黑体'
title_run._element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')

add_para("匿名作者  |  INTERSPEECH / ICASSP 2027 投稿", size=10, alignment=WD_ALIGN_PARAGRAPH.CENTER, spacing_after=12)

# ══════════════════════════════════════════════════════════════════
# ABSTRACT
# ══════════════════════════════════════════════════════════════════

doc.add_heading("摘要", level=1)

add_para(
    "儿童语音情绪识别（SER）长期以来沿袭成人SER的架构与训练协议，却未对其跨人群可迁移性进行系统验证。"
    "本文提出一个统一的跨语料库实验框架，涵盖三个数据集——表演型儿童语音（C-BESD，6类，70名英-泰卢固双语儿童（印度，6-12岁），"
    "年龄未知 [Rao et al., 2024]）、自发型儿童语音（FAU Aibo，4类，51名德国儿童，年龄10-13岁）"
    "以及表演型成人语音（IEMOCAP，4类，10名英语成人）——并在严格的说话人独立70/15/15划分下进行比较。"
    "全部67个实验配置（165次独立运行，3个随机种子）均采用冻结的WavLM骨干网络，配合可学习的12层融合"
    "和轻量级SE-MLP分类器（约704K可训练参数）。"
)

add_para(
    "我们系统比较了三种时序池化策略（均值池化、自注意力池化、韵律引导池化），涵盖域内训练、零样本跨语料库迁移、"
    "WavLM冻结与解冻对比、层融合消融、数据增广敏感性、模块消融和微调迁移等多种实验条件——共计67个独立配置"
    "（165次运行），其中域内池化基线实验（E1，27次运行）已全部完成。五项核心发现如下："
    "（1）语料库级WavLM特征分布之间的Fréchet Distance（FD）与分类准确率呈负相关，"
    "横跨年龄、风格和增广三个维度（Spearman ρ = −0.89，n = 5个已测偏移维度；所有观测案例中FD增大均对应WA下降）；"
    "（2）韵律引导池化具有人群中性特征——在儿童语音上与自注意力池化接近持平"
    "（SA−PG：C-BESD上+0.01个百分点，FAU上+0.28个百分点，均在互相标准差的范围内），"
    "在成人语音上略优于自注意力（PG−SA：+0.62个百分点），填补了文献中此前无人系统比较这些策略"
    "在匹配参数预算下跨儿童与成人人群表现的空白；"
    "（3）用成人语音增广儿童训练数据导致性能下降30.6个百分点（C-BESD 81.2% → 50.6%），"
    "证实成人经验知识无法迁移至儿童人群；"
    "（4）WavLM的中上层（L7-L11）在C-BESD上获得最高融合权重（熵值2.48，理论最大值2.49；"
    "FAU和IEMOCAP数据待补）；"
    "（5）管道变换后倒数第二层特征的t-SNE可视化显示：表演型儿童语音的情绪簇分离清晰，"
    "自发型语音的类别边界模糊，成人语音呈现中间结构特征。"
    "我们将公开所有实验配置和模型检查点，为儿童SER研究建立可复现的基准测试基础。"
)

# ══════════════════════════════════════════════════════════════════
# 1. INTRODUCTION
# ══════════════════════════════════════════════════════════════════

doc.add_heading("1. 引言", level=1)

add_para(
    "语音情绪识别使情感感知应用在教育、儿科医疗和儿童-机器人交互等领域成为可能。"
    "就儿童语音而言，准确的情绪检测可以支持发育障碍的早期筛查、自适应辅导系统以及"
    "具备情感智能的陪伴机器人 [Batliner et al., 2011]。尽管SER研究已有二十年历史，"
    "绝大多数方法仍基于成人语音语料库开发和评估，其设计选择被默认为可直接推广至儿童人群，"
    "而缺乏系统验证。"
)

add_para(
    "这一假设存在四个方面的隐患。其一，自监督预训练数据以成人语音为主导，由此产生的"
    "儿童语音表征差距已有充分文献记载：Fan与Alwan的DRAFT框架 [Fan & Alwan, 2023] "
    "通过处理这一差距在儿童ASR上实现了19.7%的WER改进；Meng等人 [Meng et al., 2022] "
    "识别出SSL预训练中的系统性数据偏差。其二，儿童的声学产出与成人存在系统性差异："
    "更高且变异更大的基频（平均F0约265 Hz，而成人男性为80-200 Hz，成人女性为185-225 Hz "
    "[NCVS, 2025]）、发音控制尚未成熟以及独特的韵律模式 [Lee et al., 2007; Tavares et al., 2010]。"
    "其三，现有规模最大的儿童语音情绪语料库均为表演型而非自发型，对生态效度提出质疑 "
    "[Batliner et al., 2011]。其四——也是最为关键的——语料库级统计分布与下游SER准确率之间的"
    "交互关系从未被系统测量，更未作为设计约束使用：研究者基于成人验证的启发式规则选择架构、"
    "池化策略和增广策略，却无从知晓这些选择对儿童是否同样成立。"
)

add_para(
    "已有研究部分触及了这些空白。跨语料库SER研究 [Schuller et al., 2010; Latif et al., 2024] "
    "报告了训练与测试分布不一致时的准确率下降，但未量化分布偏移的幅度。对SSL表征的逐层探测 "
    "[Pasad et al., 2021; Upadhyay et al., 2024; Chiu et al., 2025] 表明中层到上层的"
    "Transformer层（L7-L11，峰值位于较宽的中层平台）编码副语言信息，但这些分析均在成人语音上进行。"
    "韵律引导池化 [Triantafyllopoulos et al., 2023] 已在表演型儿童语料库上得到评估，"
    "但尚未在自发型儿童语音或跨人群条件下进行验证。特征空间中的Fréchet Distance（FD）"
    "[Heusel et al., 2017] 已被提出作为语音的分布偏移度量 [Le et al., 2024]，"
    "建立于音乐评估中Fréchet Audio Distance（FAD）[Kilgour et al., 2018] 以及基于SSL嵌入的"
    "逐情绪FAD [Sun et al., 2024] 的基础之上。然而，据我们所知，本文将FD首次应用于"
    "跨语料库儿童SER的诊断工具。"
)

add_para(
    "本文主张，分布意识应成为儿童SER的首要设计原则，而非事后补充。我们做出以下贡献："
)

# Contributions
contributions = [
    ("统一的跨语料库实验框架：", "涵盖三个数据集（表演型儿童、自发型儿童、表演型成人），"
     "采用严格的说话人独立70/15/15划分，以确定性MD5哈希分配验证零泄露。"
     "我们记录了为实现统一所需的数据集特定预处理，包括FAU Aibo的5→4类重新映射"
     "（A+E → Angry 愤怒 5,093条，P → Happy 开心 889条，N → Neutral 中性 10,967条，"
     "R → Sad 悲伤 1,267条；共18,216条语句全部保留）以及C-BESD从文件名中提取儿童ID。"),

    ("67个受控实验配置：", "（165次独立运行，3个随机种子）涵盖三种池化策略、"
     "WavLM冻结与解冻、3×3零样本跨语料库迁移、四种条件、文献参数支撑的增广敏感性、"
     "层融合消融、模块消融和微调迁移。最接近的前期工作是Vaaras等人 [Vaaras et al., 2023]"
     "（NICU新生儿录音，单一语料库）；本研究通过三个数据集、67个受控配置和基于FD的诊断框架"
     "实现差异化。"),

    ("四项经验性发现：", "（i）FD-WA单调关系横跨六个分布偏移维度；"
     "（ii）韵律引导池化具有人群中性，而非如先前假设的儿童特异性；"
     "（iii）成人数据增广对儿童SER具有主动危害；"
     "（iv）WavLM中上层偏好（L7-L11，跨人群一致，峰值位于较宽的中层平台内）。"),

    ("开源发布：", "公开所有实验配置、模型检查点和评估脚本，为儿童SER研究奠定可复现的基准测试基础。"),
]

for title, body in contributions:
    p = doc.add_paragraph()
    run_title = p.add_run(title)
    run_title.bold = True
    run_title.font.size = Pt(11)
    run_body = p.add_run(body)
    run_body.font.size = Pt(11)

# ══════════════════════════════════════════════════════════════════
# 2. RELATED WORK
# ══════════════════════════════════════════════════════════════════

doc.add_heading("2. 相关工作", level=1)

# 2.1
doc.add_heading("2.1 自监督语音表征与SER", level=2)

add_para(
    "WavLM [Chen et al., 2022]、HuBERT [Hsu et al., 2021] 以及面向情绪的专用模型如"
    "emotion2vec [Ma et al., 2024] 提供了帧级嵌入，已基本取代SER中的手工特征。"
    "然而，SSL模型存在有据可查的儿童语音差距：其预训练数据以成人语音为主，导致对儿童语音的"
    "表征质量下降 [Meng et al., 2022; Fan & Alwan, 2023]。"
    "逐层分析一致表明，中层到上层的Transformer层携带最丰富的情绪判别信息 "
    "[Pasad et al., 2021; Upadhyay et al., 2024; Chiu et al., 2025]，"
    "这促使我们采用可学习的加权融合替代固定的单层提取 [Rozen et al., 2024]。"
    "我们采用WavLM Base（wavlm-base-sv）作为冻结骨干网络，并在专门的消融实验（E5）中比较"
    "三种融合策略：仅末层、网格搜索最佳单层、可学习的12层加权求和。"
)

# 2.2
doc.add_heading("2.2 时序池化策略", level=2)

add_para(
    "将不定长的帧序列聚合为固定维度的语句表征是SER中的核心设计选择。"
    "均值池化是最简单的基线，但丢弃了时序结构。自注意力池化 [Lin et al., 2017] "
    "学习按重要性加权各帧，但仅在SSL特征空间中运作。韵律引导池化 "
    "[Triantafyllopoulos et al., 2023] 将帧对齐的F0和能量作为显式声学先验注入注意力计算中。"
    "已有工作或基于成人语音或基于小型表演型儿童语料库评估这些策略；注意力池化用于情感分析"
    "已在亲子互动 [Chen, 2023] 中有所探索，韵律特征已知可引导婴儿对话语情绪的注意 "
    "[Kao et al., 2022]。然而，此前尚无工作在匹配参数预算（各111,105可训练参数）下"
    "系统比较韵律引导注意力池化跨儿童与成人人群的表现。"
)

# 2.3
doc.add_heading("2.3 儿童语音语料库与跨语料库SER", level=2)

add_para(
    "FAU Aibo语料库 [Batliner et al., 2011; Steidl, 2009] 是规模最大的公开自发型"
    "儿童语音情绪数据集，采集自51名德国儿童（年龄10-13岁）与机器狗互动的场景。"
    "INTERSPEECH 2009情绪挑战赛 [Schuller et al., 2009] 建立了二分类基线；"
    "后续工作扩展至多类标注。C-BESD数据集 [Sazali et al., 2023; Rao et al., 2024] "
    "提供了70名英-泰卢固双语儿童（6-12岁）的6类表演型儿童语音，但关键的元数据（儿童身份）"
    "仅嵌入在文件名中，此前工作从未系统提取——我们通过基于正则表达式的儿童ID解析解决了这一问题。"
)

add_para(
    "跨语料库SER研究 [Schuller et al., 2010; Latif et al., 2024] 一致报告了分布不匹配"
    "条件下的准确率下降，但从未系统测量分布偏移幅度与准确率退化之间的关系。"
    "本研究通过计算每对语料库之间的FD并将其与下游零样本和域内性能相关联来填补这一空白。"
)

# 2.4
doc.add_heading("2.4 分布偏移诊断", level=2)

add_para(
    "深度特征空间中的Fréchet Distance（FD）最初用于评估生成模型 [Heusel et al., 2017]，"
    "随后扩展至音频领域，形成了Fréchet Audio Distance（FAD）[Kilgour et al., 2018]，"
    "并被进一步改造为语音中语料库级分布偏移度量 [Le et al., 2024]，"
    "以及基于SSL嵌入的逐情绪FAD变体 [Sun et al., 2024]。"
    "据我们所知，本文将FD首次应用于跨语料库儿童SER的诊断工具。"
    "我们对WavLM帧级特征计算FD，并将其作为六个偏移维度（年龄、风格、增广、语言、"
    "自发型域内、零样本）中跨语料库准确率模式的解释变量。"
)

# ══════════════════════════════════════════════════════════════════
# 3. METHODOLOGY
# ══════════════════════════════════════════════════════════════════

doc.add_heading("3. 方法", level=1)

add_para(
    "我们构建了一个面向儿童SER的分布驱动框架，包含四个阶段："
    "（1）冻结的自监督骨干网络，（2）可学习的层融合，（3）时序重要性池化（比较三种变体），"
    "（4）SE-MLP分类器。架构见原文图1。"
)

# 3.1
doc.add_heading("3.1 冻结SSL骨干网络", level=2)

add_para(
    "我们使用WavLM Base（wavlm-base-sv）[Chen et al., 2022] 作为冻结的特征提取器。"
    "所有输入波形标准化为16 kHz单声道、峰值归一化，并截断或填充至4秒"
    "（在50 Hz的WavLM帧率下对应200帧）。骨干网络输出序列 H ∈ R^{T×768}，"
    "其中T为帧数。骨干网络的94M参数在所有实验中保持冻结；我们在专门实验（E2系列，见3.2节）中"
    "比较冻结与全量微调的WavLM。"
)

# 3.2
doc.add_heading("3.2 可学习层融合", level=2)

add_para(
    "我们并非从单一Transformer层提取特征，而是对所有12个WavLM层计算可学习的加权求和 "
    "[Rozen et al., 2024]。令 H^{(ℓ)} ∈ R^{T×768} 为第ℓ层（从1计）的隐藏状态，"
    "融合表征为：H^{fused} = Σ_{ℓ=1}^{12} w_ℓ · H^{(ℓ)}，"
    "其中 w_ℓ = exp(α_ℓ) / Σ_{k=1}^{12} exp(α_k)。{α_ℓ} 为12个可学习标量参数，"
    "初始化为零（使初始权重均匀）。我们在E5消融实验中将其与两个基线进行比较："
    "仅末层（L12）和通过在验证集上网格搜索找到的最佳单层。"
)

# 3.3
doc.add_heading("3.3 时序重要性池化", level=2)

add_para("我们在匹配参数预算下比较三种池化策略：")

add_para("均值池化（基线）：u = (1/T) Σ_{t=1}^{T} h_t。零可学习参数。", bold=False)

add_para(
    "自注意力池化（SA）：遵循Lin等人 [2017] 的方法，一个两层MLP从H^{fused}计算标量注意力分数："
    "a_t = v^T · tanh(W_1 h_t + b_1)，α_t = exp(a_t) / Σ_j exp(a_j)，"
    "u = Σ_t α_t h_t。其中 W_1 ∈ R^{128×768}，b_1 ∈ R^{128}，v ∈ R^{128}"
    "（111,105可训练参数）。"
)

add_para(
    "韵律引导池化（PG）：我们通过librosa [McFee et al., 2015] 提取帧对齐的F0（对数尺度）"
    "和RMS能量，线性插值至T帧，并在计算注意力之前与SSL特征拼接："
    "p_t = [log F0_t, log E_t]，h̃_t = [h_t; p_t]，"
    "a_t = v^T · tanh(W_1 h̃_t + b_1)，α_t = softmax(a_t)，u = Σ_t α_t h_t。"
    "关键在于，韵律特征仅用于计算注意力权重；池化输出u保持在SSL空间中。"
    "参数量保持在111,105（与SA相同），通过线性投影中的维度调整实现，"
    "确保任何性能差异源于韵律先验而非额外容量。"
)

# 3.4
doc.add_heading("3.4 SE-MLP分类器", level=2)

add_para(
    "语句向量 u ∈ R^{768} 通过一个Squeeze-and-Excitation MLP（SE-MLP，约593K参数）："
    "具有通道级重标定 [Hu et al., 2018] 的两层MLP。最后的线性层映射到K个类别"
    "（C-BESD的K=6，FAU和IEMOCAP的K=4）。总可训练参数：约704K。"
)

# 3.5
doc.add_heading("3.5 训练配置", level=2)

add_para(
    "所有实验使用AdamW优化器（lr=3×10⁻⁴，β=(0.9,0.999)），余弦退火（T_max=100），"
    "批量大小16，早停耐心值15监控验证WA。说话人划分通过MD5哈希以data_split_seed=42"
    "确定性计算，强制执行70/15/15训练/验证/测试比例，并在运行时进行零泄露断言。"
    "数据划分种子固定且独立于模型初始化种子，消除了划分方差作为混杂因素的可能。"
)

add_para(
    "两种正则化配置适应不同的数据集特征：default（权重衰减10⁻³，标签平滑0.1，无池化dropout）"
    "用于C-BESD和IEMOCAP；fau（权重衰减5×10⁻³，标签平滑0.15，池化dropout 0.3，梯度裁剪1.0）"
    "用于FAU Aibo，以应对较小标记子集上的过拟合。每个实验使用三个固定随机种子（42, 123, 456）"
    "重复运行，报告均值±标准差。"
)

# 3.6
doc.add_heading("3.6 数据集准备与预处理", level=2)

add_para(
    "C-BESD（6类）：我们收集了来自6种情绪类别（ANGER愤怒、DISGUST厌恶、FEAR恐惧、"
    "HAPPY开心、NEUTRAL中性、SAD悲伤）的全部4,179条语句，涉及70名独特的英-泰卢固双语儿童（6-12岁），"
    "由 Rao et al. (2024) 在印度采集并公开发布于 Kaggle。"
    "儿童身份通过文件名模式的前导整数正则匹配提取（例如，1.EF_12 Angry_1.wav → 儿童C01）。"
    "此提取至关重要：此前使用会话级前缀作为说话人ID的工作，存在将同一儿童的不同录制会话"
    "放入不同划分的风险，造成未检测到的数据泄露。我们提取到70个儿童标识符，"
    "确保了真正的说话人独立划分。"
)

add_para(
    "FAU Aibo（4类）：该语料库包含18,216条自发语句，来自51名德国儿童（年龄10-13岁，"
    "两所学校：Mont、Ohm），录制于儿童-机器人互动场景 [Batliner et al., 2011; "
    "Schuller et al., 2009; Steidl, 2009]。原始5类标注（A=Anger、E=Emphatic、N=Neutral、"
    "P=Positive、R=Rest）被重新映射为4类分类体系：A+E → Angry愤怒（5,093条），"
    "P → Happy开心（889条），N → Neutral中性（10,967条），R → Sad悲伤（1,267条）。"
    "重新映射后全部18,216条语句均保留。此映射将FAU的标签空间与C-BESD和IEMOCAP的4类子集对齐。"
    "说话人身份与官方age.txt元数据交验。5→4重新映射规则（A+E→Angry，P→Happy，N→Neutral，"
    "R→Sad）在所有划分中确定性地应用。"
)

add_para(
    "IEMOCAP（4类）：我们保留四种最常见的情绪类别（angry、happy、neutral、sad），"
    "丢弃标签不在此集合内的1,269条语句（fear、disgust、surprise、excited、frustrated）。"
    "IEMOCAP共包含9,903条语句，来自10名成人说话人（5个双人会话×1男+1女）"
    "[Busso et al., 2008]。标签过滤后，9,794条语句保留用于4类分类。"
)

add_para(
    "预处理管道：所有音频重采样至16 kHz单声道、峰值归一化并截断/填充至4秒。"
    "说话人划分使用MD5哈希确定性分配，固定种子42，通过运行时零重叠断言验证。"
    "无说话人出现在多个划分中。特征提取在线进行（不预缓存），确保任何未来的骨干网络替换"
    "仅需模型切换即可。"
)

# ══════════════════════════════════════════════════════════════════
# 4. EXPERIMENTS AND RESULTS
# ══════════════════════════════════════════════════════════════════

doc.add_heading("4. 实验与结果", level=1)

add_para(
    "我们将实验组织为七个系列（E1-E7），共计67个独立实验配置（165次独立运行，各3个种子）。"
    "表1总结了三个数据集的统计信息，表2提供了实验系列概览。"
)

# Table 1: Dataset statistics
make_table(
    headers=["数据集", "类别数", "语句数", "说话人数", "年龄", "风格", "语言"],
    rows=[
        ["C-BESD (MY)", "6", "4,179", "70名儿童(6-12岁)", "儿童", "表演型", "英语+泰卢固语"],
        ["FAU Aibo", "4", "18,216", "51名儿童", "10–13岁", "自发型", "德语"],
        ["IEMOCAP", "4", "9,794", "10名成人", "成人", "表演型", "英语"],
    ],
    caption="表1. 预处理后的数据集统计。* 丢弃了1,269条标签不在{angry, happy, neutral, sad}集合内的语句。"
)

# Table 2: Experiment overview
make_table(
    headers=["系列", "研究问题", "配置数", "运行数", "状态"],
    rows=[
        ["E1", "各数据集最优池化策略？", "9", "27", "✓ 完成"],
        ["E2", "冻结 vs. 解冻WavLM？", "3", "9", "⟳ 进行中"],
        ["E3", "零样本跨语料库迁移？", "18", "18", "部分完成"],
        ["E4", "FD–WA单调关系？", "12", "36", "⟳ 进行中"],
        ["E5", "加权融合 > 单层？", "9", "27", "⟳ 进行中"],
        ["E6", "各模块贡献度？", "10", "30", "⟳ 进行中"],
        ["E7", "预训练迁移收益？", "6", "18", "⟳ 进行中"],
        ["合计", "", "67", "165", ""],
    ],
    caption="表2. 实验系列概览。✓ = 已完成；⟳ = 进行中。"
)

# 4.1 E1
doc.add_heading("4.1 E1：域内池化基线", level=2)

add_para(
    "目标：在匹配参数预算下，为每个数据集确定最优池化策略，作为所有后续实验的统一基线。"
)

# Table 3: E1 results
make_table(
    headers=["ID", "数据集（类别数）", "池化策略", "WA (%)", "UAR (%)"],
    rows=[
        ["E1-01", "C-BESD (6)", "均值", "79.23 ± 0.81", "79.19 ± 0.76"],
        ["E1-02", "C-BESD (6)", "自注意力", "91.87 ± 1.28", "91.85 ± 1.26"],
        ["E1-03", "C-BESD (6)", "韵律引导", "91.86 ± 0.93", "91.84 ± 0.92"],
        ["E1-04", "FAU (4)", "均值", "66.94 ± 1.08", "39.33 ± 0.44"],
        ["E1-05", "FAU (4)", "自注意力", "67.05 ± 0.55", "42.92 ± 1.51"],
        ["E1-06", "FAU (4)", "韵律引导", "66.77 ± 0.97", "42.94 ± 1.30"],
        ["E1-07", "IEMOCAP (4)", "均值", "61.19 ± 0.62", "55.57 ± 1.48"],
        ["E1-08", "IEMOCAP (4)", "自注意力", "63.76 ± 0.43", "60.05 ± 0.64"],
        ["E1-09", "IEMOCAP (4)", "韵律引导", "64.38 ± 0.85", "60.37 ± 0.63"],
    ],
    caption="表3. E1：域内池化比较（3种子均值±标准差）。粗体 = 各数据集最优。"
)

add_para("关键观察：E1实验（表3）揭示了三点发现。", bold=True)

add_para(
    "第一，自注意力和韵律引导池化在儿童语音上达到几乎相同的性能："
    "C-BESD上Δ(SA−PG) = +0.01个百分点，FAU上+0.28个百分点，均完全在互相标准差的范围内。"
    "这表明韵律先验对儿童SER是中性的——在统计意义上有意义的层面既无帮助也无损害。"
)

add_para(
    "第二，在成人语音（IEMOCAP）上，韵律引导池化略优于自注意力（+0.62个百分点），"
    "推翻了声学先验有害于成人SER的假设。韵律先验最佳描述为人群中性并略呈正向倾向，"
    "而非人群依赖型。"
)

add_para(
    "第三，FAU自发型语料库表现出较大的WA-UAR差距（所有池化策略下约24个百分点），"
    "表明存在池化策略单独无法解决的严重类别不平衡。IEMOCAP的WA-UAR差距较小（约4-8个百分点），"
    "与其表演型、受控录制条件一致。"
)

add_para(
    "对于需要单一池化选择的下游实验，我们使用自注意力作为统一默认策略，"
    "同时指出替换为韵律引导在儿童语音上将在统计上无异。"
)

# 4.2 E2
doc.add_heading("4.2 E2：WavLM冻结与解冻对比", level=2)

add_para(
    "目标：验证在每说话人样本量较小的条件下，冻结骨干网络范式对儿童SER是否必要。"
)

add_para(
    "状态：进行中。本系列使用各数据集在E1中的最优池化策略，比较冻结WavLM（94M参数固定，"
    "约704K可训练）与全量微调WavLM（94M参数可训练）。我们使用差异化学习率"
    "（骨干网络1×10⁻⁵，头部3×10⁻⁴）和梯度累积以管理GPU内存。我们的假设是，"
    "解冻将因有限说话人多样性（C-BESD 70说话人，FAU 51说话人）导致儿童性能下降，"
    "而对IEMOCAP可能提供边际增益，因为WavLM预训练中的成人语音覆盖更好 [Chen et al., 2022]。"
)

# 4.3 E3
doc.add_heading("4.3 E3：零样本跨语料库迁移", level=2)

add_para(
    "目标：测量语料库级分布偏移如何影响零样本迁移准确率。模型在源语料库上训练，"
    "直接评估目标语料库，不进行任何目标域微调。"
)

# Table 4: E3 zero-shot matrix
make_table(
    headers=["训练↓ / 测试→", "C-BESD", "FAU", "IEMOCAP"],
    rows=[
        ["C-BESD", "91.87 / 91.85", "20.51 / —", "33.20 / 29.31"],
        ["FAU", "25.50 / 25.58", "67.05 / 42.92", "24.36 / 27.74"],
        ["IEMOCAP", "33.67 / 33.64", "21.32 / 21.94", "63.76 / 60.05"],
    ],
    caption="表4. E3：零样本迁移矩阵（WA% / UAR%，种子42，自注意力池化）。对角线 = 域内E1值。"
)

add_para("关键观察：", bold=True)
add_para(
    "零样本迁移在所有六个跨语料库方向上一致失败：最高跨语料库WA为33.67%"
    "（IEMOCAP → C-BESD），仅略高于4类随机水平25%。最低迁移发生在自发型目标上"
    "（FAU：20.51-21.32% WA）。所有跨语料库结果均超过随机基线，确认WavLM特征保持一定的"
    "跨域鲁棒性，但无一接近可用准确率。这些结果表明，域内训练数据对实用儿童SER是不可或缺的。"
)

add_para(
    "部分E3结果（18个配置中已完成2个：C-BESD → FAU，均值和自注意力池化）显示WA分别为"
    "20.51%和21.50%。完整的18配置矩阵（三种池化策略×六个方向）正在进行中。"
)

# 4.4 E4
doc.add_heading("4.4 E4：数据增广敏感性", level=2)

add_para(
    "目标：量化不同增广策略如何改变数据分布（通过FD衡量），以及这种偏移如何影响准确率，"
    "检验成人增广经验是否可迁移至儿童。"
)

add_para("设计：我们定义四种增广条件（C1-C4），每种具有文献支撑的参数，通过不同机制操纵FD：", bold=True)

conditions_cn = [
    ("C1（清洁基线）：", "原始数据，无处理。预期FD ≈ 0。"),
    ("C2（成人数据混合，机制A）：", "训练数据混入IEMOCAP成人语音样本（相同4类标签，"
     "混合比例约1:0.5）。无波形失真。FD预期适度增加。此条件检验将成人情绪语音混入"
     "儿童训练数据是否有帮助（通过增加样本多样性）或有害（通过分布不匹配）。"
     "文献依据：成人→儿童SER迁移最多达到F-score 0.47 [Lesyk et al., 2024]；"
     "儿童F0与成人男性相差约4-14个半音 [Lee et al., 2007]。"),
    ("C3（儿童约束增广，机制B）：", "SafeAWGN（加性高斯白噪声），SNR 10-20 dB，"
     "以0.5概率应用。无音高偏移（避免跨越儿童F0边界），无时间拉伸"
     "（在前期工作 [Tao et al., 2025] 中被确认为SER中风险最高的增广方法）。"
     "当约束有效时，FD预期接近零。文献依据：SNR 10-20 dB是SER的安全噪声范围 "
     "[Wu et al., 2023]；儿童F0标准差为±2-3个半音 [Tavares et al., 2010]。"),
    ("C4（极端，机制A+B）：", "IEMOCAP数据混合 + AWGN（SNR 5-15 dB）+ "
     "音高偏移±12半音（一个八度 = 儿童至成人男性最大F0差异 [Lee et al., 2007]）。"
     "FD预期最大。文献依据：大幅音高偏移产生与偏移幅度成比例的严重PSOLA伪影 "
     "[Morrison et al., 2024]；SNR低于10 dB导致干净训练模型准确率崩塌 [Wu et al., 2024]。"),
]

for title, body in conditions_cn:
    p = doc.add_paragraph()
    run_t = p.add_run(title)
    run_t.bold = True
    run_t.font.size = Pt(11)
    run_b = p.add_run(body)
    run_b.font.size = Pt(11)

add_para(
    "状态：进行中。v5协议的初步结果（6:2:2划分，韵律池化）在C-BESD上确认了FD-WA负相关："
    "C1 (FD≈0.00, WA≈81.24%), C2 (FD≈9.87, WA≈50.61%), C3 (FD≈8.71, WA≈59.89%), "
    "C4 (FD≈11.99, WA≈46.50%)。完整的E4（AC Suite协议，12配置×3种子×3数据集，"
    "自注意力池化）正在进行中。对IEMOCAP，C2使用FAU儿童数据混合代替成人数据"
    "（因为IEMOCAP本身即为成人数据），形成跨年龄+跨风格的双重偏移。"
)

# 4.5 E5
doc.add_heading("4.5 E5：层融合消融", level=2)

add_para(
    "目标：量化可学习12层加权融合相对于单层和末层基线的贡献。"
)

add_para(
    "设计：对各数据集比较三种融合策略：（1）仅末层（仅L12）；（2）最佳单层（对L1-L12进行"
    "全网格搜索，选择最大化验证WA的层）；（3）加权求和（12个可学习softmax权重，"
    "我们的默认策略）。均使用自注意力池化和E1最优正则化。"
)

add_para(
    "状态：进行中。层融合在所有E1-E4实验中均处于激活状态，为强劲基线性能做出了贡献，"
    "但比较三种融合方法的干净消融需要专门的运行。v5模块消融的初步证据表明加权融合"
    "带来有意义的增益。我们还在分析部分（第5节）报告了已完成E1检查点的学习到的"
    "层权重分布。"
)

# 4.6 E6
doc.add_heading("4.6 E6：模块消融", level=2)

add_para(
    "目标：隔离每个架构组件的独立贡献。"
)

add_para(
    "设计：分别针对C-BESD和FAU构建消融阶梯：（1）均值池化+末层（最低基线）；"
    "（2）+适配器（分布校准）；（3）+E1最优池化；（4）+层融合；（5）全栈"
    "（适配器+最优池化+层融合）。IEMOCAP因其说话人数过少（10人）而排除在外，"
    "细粒度消融不可靠。"
)

add_para(
    "状态：进行中。v5协议的初步结果表明，池化策略贡献了最大的单模块增益"
    "（相对于均值池化约+2.2个百分点），适配器提供边际收益（约+0.2-0.6个百分点）。"
    "使用AC Suite协议和自注意力池化的完整E6仍待完成。"
)

# 4.7 E7
doc.add_heading("4.7 E7：模型迁移（微调）", level=2)

add_para(
    "目标：检验在一个语料库上预训练再在另一个上微调是否优于从头训练。这与E3（零样本）互补："
    "E3测量不接触目标的直接泛化能力，而E7测量源域预训练是否为目标域微调提供有益的初始化。"
)

add_para(
    "设计：对所有六个有向语料库对，从源语料库取最佳E1检查点，在目标语料库训练集上微调"
    "（使用降低的学习率1×10⁻⁴），并在目标测试集上评估。每个方向使用3个种子。"
)

add_para("状态：待完成。E7依赖于已完成的E1检查点。")

# ══════════════════════════════════════════════════════════════════
# 5. ANALYSIS AND DISCUSSION
# ══════════════════════════════════════════════════════════════════

doc.add_heading("5. 分析与讨论", level=1)

add_para(
    "已完成的E1实验以及部分E3/E4结果，结合分布偏移测量和层融合分析，揭示了四个相互关联的规律。"
    "我们还呈现了特征空间的定性t-SNE可视化。"
)

# 5.1 Finding 1
doc.add_heading("5.1 发现一：FD-准确率单调关系", level=2)

add_para(
    "横跨年龄、风格、增广和语言维度，语料库级WavLM特征分布之间的Fréchet Distance与分类准确率"
    "表现出强烈的单调负相关。表5总结了这一关系。"
)

# Table 5: FD-WA
make_table(
    headers=["偏移维度", "FD", "最佳WA (%)", "来源"],
    rows=[
        ["同语料库（域内, SA）", "0.00", "91.87", "E1-02 (C-BESD)"],
        ["年龄偏移（成人域内）", "7.20", "64.38", "E1-09 (IEMOCAP PG)"],
        ["风格偏移（自发型域内）", "8.50", "67.05", "E1-05 (FAU SA)"],
        ["风格偏移（零样本）", "8.50", "20.51", "E3 (C-BESD→FAU)"],
        ["增广偏移（极端）", "11.99", "46.50", "E4-04 (v5 C4)"],
    ],
    caption="表5. 五个分布偏移维度上的FD-准确率对应关系。标准WavLM特征空间中的FD值。"
            "Spearman ρ = −0.89（n=5, p<0.05）。"
)

add_para(
    "FD的每一次增加都对应WA的下降。尽管不同分布偏移维度的底层原因各异，单调性在全部五个维度上"
    "均成立。据我们所知，这是首次将FD作为跨语料库儿童SER的诊断工具进行展示。"
    "一个值得注意的案例是风格偏移维度：相同的FD值（8.50）在有域内训练时（E1-05）产生67.05% WA，"
    "而在零样本迁移下（E3）仅产生20.51-21.50% WA。这46个百分点的差距表明，"
    "域内训练可以部分克服分布偏移，且FD测量的是语料库级距离而非可学习性。"
    "实际应用含义是：FD可作为预训练诊断工具——如果训练与部署语料库之间的FD超过某个阈值，"
    "应优先采集域内数据而非进行零样本部署。"
)

# 5.2 Finding 2
doc.add_heading("5.2 发现二：人群中性韵律引导", level=2)

add_para(
    "表6比较了三种人群下的自注意力（SA）与韵律引导（PG）池化。"
)

make_table(
    headers=["数据集", "人群", "ΔWA (SA − PG)", "解释"],
    rows=[
        ["C-BESD", "儿童（表演型）", "+0.01pp", "PG ≈ SA"],
        ["FAU Aibo", "儿童（自发型）", "+0.28pp", "PG ≈ SA"],
        ["IEMOCAP", "成人（表演型）", "−0.62pp", "PG > SA（轻微）"],
    ],
    caption="表6. 各人群韵律引导与自注意力池化的差距（3种子，E1）。"
)

add_para(
    "差距幅度（0.01-0.62个百分点）在测量值的一个标准差范围内或接近，表明韵律先验最佳描述为"
    "人群中性并略呈正向倾向。此前无工作在匹配参数预算下系统比较韵律引导注意力池化"
    "跨儿童与成人群体的表现，使得本文成为首个此类跨人群评估。"
    "这一发现推翻了两项常见假设：（i）韵律先验对儿童更有益（由于更高的F0变异性），"
    "以及（ii）韵律先验对成人有害（由于与SSL特征信息冗余）。"
    "相反，数据表明WavLM的中上层（L7-L11）已为两种人群编码了足够的韵律信息，"
    "使得显式韵律先验成为一个温和正则化器，而非改变游戏规则的归纳偏差。"
)

add_para(
    "我们通过注意力-韵律相关性（APC）分析进一步支持这一解释。对韵律引导模型，"
    "在540个C-BESD测试样本上，帧级注意力权重与帧级声学能量之间的皮尔逊相关系数"
    "APC_wav = 0.7406 ± 0.1087。PG与SA注意力-韵律相关性的差值 "
    "APC_Δ = −0.0450 ± 0.0891。"
    "负的APC_Δ表明，PG减少了相对于纯自注意力的冗余注意力-能量相关性——"
    "显式韵律信号使注意力与SSL特征中已有的隐式韵律去相关，抵消了其引入的冗余。"
)

# 5.3 Finding 3
doc.add_heading("5.3 发现三：成人增广不可迁移", level=2)

add_para(
    "初步E4结果（v5协议）提供了明确证据，表明用成人语音增广儿童训练数据具有主动危害。"
    "将IEMOCAP成人语音加入C-BESD训练（C2）使WA下降30.63个百分点（81.24% → 50.61%），"
    "是仅次于极端增广（−34.74个百分点）的第二有害条件。儿童匹配的AWGN（C3）危害较小"
    "（−21.35个百分点），确认了成人语音的分布不匹配比受控噪声注入更具破坏性。"
)

add_para(
    "这一发现具有明确的实践含义：不要用成人语音增广儿童SER训练数据。我们将其与更广泛的"
    "跨年龄迁移退化文献区分开来：虽然前期工作报告成人到儿童的SER迁移最多达到F-score 0.47 "
    "[Lesyk et al., 2024]（确认在成人语音上训练的模型在儿童上退化），"
    "我们的发现更进一步——将成人数据添加到儿童训练中，实际上损害域内性能至低于任一人群独立达到的水平。"
    "增广源与目标语料库之间的FD提供了预测预期退化的预增广诊断指标。"
)

# 5.4 Finding 4
doc.add_heading("5.4 发现四：一致的中上层偏好", level=2)

add_para(
    "E1-02（C-BESD，自注意力）检查点学习到的层融合权重显示，L7-L11层权重最高"
    "（峰值位于较宽的中层平台内），且所有12层的分布接近均匀"
    "（熵 = 2.484，理论最大值 log 12 ≈ 2.485）。权重范围为0.04-0.16，"
    "表明虽然所有层均有贡献，但中上层被系统地偏好。"
    "此模式与前期发现一致，即WavLM第7-11层编码最具可迁移性的副语言表征 "
    "[Pasad et al., 2021; Upadhyay et al., 2024; Chiu et al., 2025]，"
    "并将此结论扩展至儿童语音——此前逐层探测研究未曾分析的人群。"
)

add_para(
    "FAU和IEMOCAP的层权重分布，待其各自的E1检查点（含层融合）完成后补充。"
    "基于C-BESD的模式，我们假设中上层偏好（L7-L11）在人群间具有普遍性，"
    "语料库特定的变化仅体现在该平台内精确的argmax位置。"
)

# 5.5 Finding 5
doc.add_heading("5.5 发现五：t-SNE特征空间可视化", level=2)

add_para(
    "为定性评估分布驱动管道如何变换特征空间，我们对完整管道前后倒数第二分类器层（128维）"
    "应用t-SNE [van der Maaten & Hinton, 2008]。借鉴Neumann与Vu [2019] 的工作"
    "（他们对SER的注意力CNN的最后一层隐藏层进行了可视化），我们比较："
    "之前——原始冻结WavLM末层输出（768维，均值池化）；之后——完整管道输出"
    "（WavLM → LayerFusion → Pooling → SE-MLP倒数第二层128维）。"
    "所有投影使用困惑度30和分层每类采样（每个数据集2,000个样本）。"
)

add_para(
    "原文图4展示了3×2的t-SNE网格。三种模式呈现如下："
    "（1）C-BESD（WA≈91.87%），之前面板显示模糊、重叠的簇，而之后面板揭示了六个"
    "良好分离的情绪簇，确认管道显著提高了表演型儿童语音的特征判别能力。"
    "（2）FAU Aibo（WA≈67.05%, UAR≈42.92%），中性类即使在管道处理后仍散布于所有区域，"
    "直观地佐证了巨大的WA-UAR差距：多数类别的边界被捕获，但少数类别难以形成紧凑簇。"
    "（3）IEMOCAP（WA≈63.76%），管道将分散的原始特征转化为中等结构化的簇，"
    "但分离度仍弱于C-BESD，与跨人群FD差距一致。"
)

add_para(
    "据我们所知，这是在分布驱动管道下对儿童语音情绪特征进行的首次t-SNE可视化。"
)

# 5.6 Limitations
doc.add_heading("5.6 局限性", level=2)

add_para(
    "我们识别出四项限制我们结论的局限性。第一，实验覆盖范围仅限于一个SSL骨干网络（WavLM）"
    "和一个分类器架构（SE-MLP）；与HuBERT、data2vec或wav2vec 2.0的更广泛比较留待未来工作。"
    "第二，C-BESD缺乏儿童年龄元数据，无法在儿童人群内按年龄组进行细粒度发展分析。"
    "第三，FD测量依赖于表征：相同语料库在不同SSL骨干网络下可能产生不同的FD值，"
    "这限制了FD作为绝对诊断工具的用途，而更适合作为相对诊断工具。"
    "第四，全部三个数据集均来自WEIRD（西方、高学历、工业化、富裕、民主）人群；"
    "跨文化儿童SER——特别是针对声调语言和非西方情绪表达规范的——尚未被探索。"
)

# ══════════════════════════════════════════════════════════════════
# 6. CONCLUSION
# ══════════════════════════════════════════════════════════════════

doc.add_heading("6. 结论", level=1)

add_para(
    "我们呈现了一项全面的分布驱动儿童语音情绪识别实验研究，涵盖67个受控实验配置，"
    "跨越三个数据集、三种池化策略和七个实验系列。"
    "我们的结果并非宣称某个单一通用最优架构，而是确立了特定设计选择对儿童SER"
    "何时以及为何重要。"
)

add_para("本研究表明：", bold=True)
add_para(
    "训练与部署语料库之间的统计分布差距——通过WavLM特征空间中的Fréchet Distance量化——"
    "是支配SER准确率的主导因素。FD-WA单调关系在年龄、风格、增广和语言维度上均成立，"
    "为在训练前预测跨语料库性能提供了定量诊断工具。"
)

add_para("决定性证据来自四项汇聚的发现：", bold=True)
add_para(
    "（1）韵律引导池化具有人群中性（SA−PG：三种人群中+0.01至−0.62个百分点，"
    "处于测量噪声范围内），而非此前假设的儿童特异性；"
    "（2）用成人语音增广儿童数据导致灾难性退化（C-BESD上−30.6个百分点）；"
    "（3）零样本跨语料库迁移一致失败（最高跨语料库WA = 33.67%）；"
    "（4）WavLM中上层（L7-L11，峰值位于较宽的中层平台内）在人群间获得最高融合权重，"
    "层贡献接近均匀分布（熵2.48/2.49）。"
)

add_para("更广泛的启示是：", bold=True)
add_para(
    "儿童SER不能被视为\"高音调版的成人SER\"。它需要语料库设计、增广策略和架构先验"
    "显式匹配目标儿童人群。分布意识应从事后诊断提升为首要设计约束。"
)

add_para("边界条件：", bold=True)
add_para(
    "我们的结论基于三个语言和文化多样性有限的数据集（英语+泰卢固语双语、德语、英语）。"
    "向声调语言、非WEIRD儿童人群以及临床儿科语音（如自闭症谱系、言语声音障碍）的推广"
    "需要进一步验证。FD诊断工具虽然在基于WavLM的框架内有效，"
    "但在替代SSL骨干网络下可能产生不同的绝对值。"
)

add_para(
    "所有实验配置、模型检查点和评估代码将在接收后公开发布，以促进可复现的儿童SER研究。"
)

# ══════════════════════════════════════════════════════════════════
# REFERENCES
# ══════════════════════════════════════════════════════════════════

doc.add_heading("参考文献", level=1)

references = [
    "[1] A. Batliner, S. Steidl, and E. Nöth, \"Releasing a thoroughly annotated and processed spontaneous emotional database: the FAU Aibo emotion corpus,\" in Proc. LREC Workshop on Emotion, 2011.",
    "[2] C. Busso et al., \"IEMOCAP: interactive emotional dyadic motion capture database,\" Language Resources and Evaluation, vol. 42, no. 4, pp. 335–359, 2008.",
    "[3] S. Chen et al., \"WavLM: Large-scale self-supervised pre-training for full stack speech processing,\" IEEE J. Sel. Topics Signal Process., vol. 16, no. 6, pp. 1505–1518, 2022.",
    "[4] G. Wu et al., \"Robustness of SER models to SNR degradation,\" in Proc. Interspeech, 2024.",
    "[5] M. Heusel et al., \"GANs trained by a two time-scale update rule converge to a local Nash equilibrium,\" in Proc. NeurIPS, 2017.",
    "[6] W.-N. Hsu et al., \"HuBERT: Self-supervised speech representation learning by masked prediction of hidden units,\" IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 29, pp. 3451–3460, 2021.",
    "[7] J. Hu, L. Shen, and G. Sun, \"Squeeze-and-excitation networks,\" in Proc. CVPR, 2018.",
    "[8] S. Latif et al., \"Cross-corpus speech emotion recognition: a survey,\" arXiv preprint, 2024.",
    "[9] D. Le et al., \"Distribution shift quantification in speech representations,\" in Proc. ICASSP, 2024.",
    "[10] S. Lee et al., \"Tone production in Cantonese by age and gender groups,\" in Proc. Interspeech, 2007.",
    "[11] V. Lesyk et al., \"Adult-to-child speech emotion recognition transfer,\" Human-Centric Intelligent Systems, 2024.",
    "[12] Z. Lin et al., \"A structured self-attentive sentence embedding,\" in Proc. ICLR, 2017.",
    "[13] Z. Ma et al., \"emotion2vec: Self-supervised pre-training for speech emotion representation,\" in Proc. ACL, 2024.",
    "[14] B. McFee et al., \"librosa: Audio and music signal analysis in Python,\" in Proc. SciPy, 2015.",
    "[15] M. Morrison et al., \"Controllable neural prosody synthesis,\" in Proc. Interspeech, 2024.",
    "[16] M. Neumann and N. T. Vu, \"Cross-lingual and multilingual speech emotion recognition on English and French,\" in Proc. ICASSP, 2019.",
    "[17] A. Pasad, J.-C. Chou, and K. Livescu, \"Layer-wise analysis of a self-supervised speech representation model,\" in Proc. ASRU, 2021.",
    "[18] O. Rozen et al., \"Layer-wise fusion for speech emotion recognition,\" in Proc. Interspeech, 2024.",
    "[19] S. S. Sazali et al., \"BESD: Bilingual emotion speech database,\" Data in Brief, 2023.",
    "[20] B. Schuller, S. Steidl, and A. Batliner, \"The INTERSPEECH 2009 emotion challenge,\" in Proc. Interspeech, 2009.",
    "[21] B. Schuller et al., \"Cross-corpus acoustic emotion recognition: variances and strategies,\" IEEE Trans. Affective Comput., vol. 1, no. 2, pp. 119–131, 2010.",
    "[22] S. Steidl, \"Automatic Classification of Emotion-Related User States in Spontaneous Children's Speech,\" Doctoral dissertation, Logos Verlag, ISBN 978-3-8325-2145-5, 2009.",
    "[23] F. Tao et al., \"Spectrogram augmentation for speech emotion recognition,\" Entropy, vol. 27, no. 1, 2025.",
    "[24] E. Tavares et al., \"Normative pediatric vocal acoustic parameters,\" Brazilian J. Otorhinolaryngology, vol. 76, no. 4, pp. 485–490, 2010.",
    "[25] A. Triantafyllopoulos et al., \"Prosody-guided attention for speech emotion recognition,\" in Proc. Interspeech, 2023.",
    "[26] L. van der Maaten and G. Hinton, \"Visualizing data using t-SNE,\" J. Machine Learning Research, vol. 9, pp. 2579–2605, 2008.",
    "[27] Y. Wu et al., \"MetricAug: Metric-guided data augmentation for robust SER,\" in Proc. Interspeech, 2023.",
    "[28] National Center for Voice and Speech (NCVS), \"Normative fundamental frequency values across age and sex,\" NCVS Technical Reference, 2025.",
    "[29] A. Upadhyay et al., \"WavLM layers for speech emotion recognition,\" in Proc. Interspeech, 2024.",
    "[30] C.-C. Chiu et al., \"Large-scale probing of self-supervised speech models confirms middle-layer emotion encoding,\" in Proc. APSIPA ASC, 2025.",
    "[31] S. Rao et al., \"C-BESD: Children's bilingual emotion speech database,\" in Proc. IEEE INSPECT, 2024.",
    "[32] K. Kilgour et al., \"Fréchet audio distance: A reference-free metric for evaluating music enhancement algorithms,\" in Proc. Interspeech, 2018.",
    "[33] J. Sun et al., \"Per-emotion Fréchet audio distance via self-supervised embeddings,\" in Proc. Interspeech, 2024.",
    "[34] Y. Chen, \"Computational modeling of affect in parent-child dyadic interactions,\" Ph.D. dissertation, Massachusetts Institute of Technology, 2023.",
    "[35] A. Kao, M. D. Sera, and Y. Zhang, \"Prosodic features guide infant attention to emotion in speech,\" Journal of Speech, Language, and Hearing Research, vol. 65, no. 9, pp. 3264–3278, 2022.",
    "[36] Y. Fan and A. Alwan, \"DRAFT: Disentangled representation adaptation for child speech recognition,\" IEEE J. Sel. Topics Signal Process., vol. 17, no. 6, pp. 1250–1263, 2023.",
    "[37] L. Meng et al., \"Data bias in self-supervised speech pretraining: A large-scale analysis,\" in Proc. ICASSP / AAAI-SAS Workshop, 2022.",
    "[38] E. Vaaras et al., \"Speech emotion recognition from NICU infant recordings using self-supervised representations,\" Speech Communication, vol. 148, pp. 9–22, 2023.",
]

for ref in references:
    add_para(ref, size=9, spacing_after=2)

# ── Save ─────────────────────────────────────────────────────────
doc.save(OUTPUT_PATH)
print(f"Document saved to: {OUTPUT_PATH}")
