"""PPT v4 Final: RGB(2,74,21) green theme, Chinese, embedded charts, no ASCII art."""
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

PROJECT = 'D:/大学/论文/儿童语音情绪识别/新方案-分布驱动儿童SER'
TEMPLATE = os.path.join(PROJECT, 'paper_draft', 'BIT_Beamer_Template.pptx')
FIG = os.path.join(PROJECT, 'paper_draft', 'figures')
OUT = os.path.join(PROJECT, 'paper_draft', 'presentation_v4.pptx')

prs = Presentation(TEMPLATE)
while len(prs.slides) > 0:
    rId = prs.slides._sldIdLst[0].get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
    prs.part.drop_rel(rId); prs.slides._sldIdLst.remove(prs.slides._sldIdLst[0])
LAYOUT = prs.slide_layouts[6]

# RGB(2,74,21) green theme
GREEN = RGBColor(2, 74, 21); GREEN_D = RGBColor(1, 50, 14); GREEN_L = RGBColor(3, 106, 30)
WHITE = RGBColor(255,255,255); BLACK = RGBColor(26,26,26); GRAY = RGBColor(119,119,136)
LGRAY = RGBColor(238,238,242); RED = RGBColor(178,24,43); GOLD = RGBColor(133,100,4)
TOTAL = 21

def ns():
    s = prs.slides.add_slide(LAYOUT)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = WHITE
    return s

def footer(s, n):
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(7.05), Inches(13.33), Inches(0.45))
    bar.fill.solid(); bar.fill.fore_color.rgb = GREEN; bar.line.fill.background()
    t1 = s.shapes.add_textbox(Inches(0.5), Inches(7.1), Inches(6), Inches(0.35))
    tf = t1.text_frame; p = tf.paragraphs[0]
    p.text = '分布驱动儿童语音情绪识别  |  INTERSPEECH / ICASSP 2027'
    p.font.size = Pt(9); p.font.color.rgb = WHITE
    t2 = s.shapes.add_textbox(Inches(10.5), Inches(7.1), Inches(2.5), Inches(0.35))
    tf = t2.text_frame; p = tf.paragraphs[0]
    p.text = f'{n} / {TOTAL}'; p.font.size = Pt(9); p.font.color.rgb = WHITE; p.alignment = PP_ALIGN.RIGHT

def title_bar(s, text):
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.33), Inches(0.85))
    bar.fill.solid(); bar.fill.fore_color.rgb = GREEN; bar.line.fill.background()
    t = s.shapes.add_textbox(Inches(0.6), Inches(0.1), Inches(12), Inches(0.65))
    tf = t.text_frame; p = tf.paragraphs[0]
    p.text = text; p.font.size = Pt(24); p.font.bold = True; p.font.color.rgb = WHITE

def body(s, lines, top=1.2, left=0.6, width=12, size=14):
    t = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(5.5))
    tf = t.text_frame; tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line; p.font.size = Pt(size); p.font.color.rgb = BLACK; p.space_after = Pt(5)

def two_col(s, left_lines, right_lines, top=1.2):
    for lpos, lines in [(0.5, left_lines), (6.8, right_lines)]:
        t = s.shapes.add_textbox(Inches(lpos), Inches(top), Inches(5.8), Inches(5.3))
        tf = t.text_frame; tf.word_wrap = True
        for i, line in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = line; p.font.size = Pt(12); p.font.color.rgb = BLACK; p.space_after = Pt(4)

def section_slide(s, num, title, subtitle=''):
    s.background.fill.solid(); s.background.fill.fore_color.rgb = GREEN_D
    t1 = s.shapes.add_textbox(Inches(1.5), Inches(2.2), Inches(10), Inches(1))
    tf = t1.text_frame; p = tf.paragraphs[0]
    p.text = f'{num}.  {title}'; p.font.size = Pt(36); p.font.bold = True; p.font.color.rgb = WHITE
    if subtitle:
        t2 = s.shapes.add_textbox(Inches(1.5), Inches(3.5), Inches(10), Inches(0.6))
        tf2 = t2.text_frame; p2 = tf2.paragraphs[0]
        p2.text = subtitle; p2.font.size = Pt(16); p2.font.color.rgb = RGBColor(0xCC,0xDD,0xCC)

def img(s, path, left, top, width, height=None):
    h = height if height else width * 0.6
    if os.path.exists(path):
        s.shapes.add_picture(path, Inches(left), Inches(top), Inches(width), Inches(h))
    else:
        shape = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(h))
        shape.fill.solid(); shape.fill.fore_color.rgb = LGRAY; shape.line.color.rgb = GRAY; shape.line.width = Pt(0.5)
        tf = shape.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.text = f'[图表: {os.path.basename(path)}]'; p.font.size = Pt(12); p.font.color.rgb = GRAY; p.alignment = PP_ALIGN.CENTER

def table_slide(s, headers, rows, top=1.3, left=0.5):
    nrows = len(rows) + 1; ncols = len(headers)
    tbl_shape = s.shapes.add_table(nrows, ncols, Inches(left), Inches(top), Inches(12.3), Inches(0.42 * nrows))
    tbl = tbl_shape.table
    for j, h in enumerate(headers):
        c = tbl.cell(0, j); c.text = h
        for p in c.text_frame.paragraphs: p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = WHITE
        c.fill.solid(); c.fill.fore_color.rgb = GREEN
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = tbl.cell(i+1, j); c.text = str(v)
            for p in c.text_frame.paragraphs: p.font.size = Pt(10); p.font.color.rgb = BLACK
            if i % 2 == 1: c.fill.solid(); c.fill.fore_color.rgb = LGRAY

pn = 0

# ===== 1: TITLE =====
s = ns(); pn += 1
s.background.fill.solid(); s.background.fill.fore_color.rgb = GREEN_D
t1 = s.shapes.add_textbox(Inches(1), Inches(1.5), Inches(11), Inches(1.5))
tf = t1.text_frame; p = tf.paragraphs[0]
p.text = '分布驱动儿童语音情绪识别'; p.font.size = Pt(40); p.font.bold = True; p.font.color.rgb = WHITE
t2 = s.shapes.add_textbox(Inches(1), Inches(3.0), Inches(11), Inches(1.2))
tf = t2.text_frame
p1 = tf.paragraphs[0]; p1.text = 'Distribution-Driven Child Speech Emotion Recognition'; p1.font.size = Pt(20); p1.font.color.rgb = RGBColor(0xCC,0xDD,0xCC)
p2 = tf.add_paragraph(); p2.text = '3个数据集 · 7个实验系列 · 63组对照实验'; p2.font.size = Pt(16); p2.font.color.rgb = RGBColor(0xAA,0xBB,0xAA)
t3 = s.shapes.add_textbox(Inches(1), Inches(5.0), Inches(11), Inches(0.8))
tf = t3.text_frame; p = tf.paragraphs[0]
p.text = 'INTERSPEECH / ICASSP 2027  |  2026年6月'; p.font.size = Pt(14); p.font.color.rgb = RGBColor(0x99,0xAA,0x99)
footer(s, pn)

# ===== 2: OUTLINE =====
s = ns(); pn += 1; title_bar(s, '目录')
body(s, ['1. 研究动机 — 为什么需要分布驱动的儿童语音情绪识别',
    '2. 核心贡献 — 四项可复现的工程规律','3. 数据集与预处理 — 三个语料库的统一流水线',
    '4. 模型架构 — WavLM + 层融合 + 三种池化策略',
    '5. 实验设计 — 7个系列、63组对照实验','6. 主实验结果 — 域内基线、零样本矩阵、消融',
    '7. 核心发现 — FD单调性、韵律特异性、增强不可迁移、层融合','8. 总结与展望'], top=1.3); footer(s, pn)

# ===== SECTION 1 =====
s = ns(); pn += 1; section_slide(s, 1, '研究动机', '为什么需要分布驱动的儿童语音情绪识别？'); footer(s, pn)

s = ns(); pn += 1; title_bar(s, '儿童语音情绪识别的核心矛盾')
two_col(s, [
    '① SSL预训练偏差',
    '  WavLM/HuBERT预训练数据几乎全部是成人语音',
    '  （LibriSpeech + GigaSpeech + VoxPopuli）',
    '  → 儿童声学特征(F0 250-330Hz vs 成人~120Hz)未被充分表示',
    '',
    '② 语料库风格鸿沟',
    '  演绎式：C-BESD、IEMOCAP — 标准化录音、刻意表达',
    '  自发性：FAU Aibo — 机器人交互、噪声环境、自然表达',
    '  → 同样架构：C-BESD 92.78% → FAU 19.56%（零样本崩塌73pp）',
], [
    '③ 定量诊断工具缺失',
    '  过去没有统一方法来测量语料库间的分布偏移',
    '  → 我们引入Fréchet Distance (FD)',
    '  → 在WavLM特征空间中定量测量语料库级分布距离',
    '',
    '④ 儿童数据稀缺性',
    '  C-BESD：4,179条（70名儿童）',
    '  FAU Aibo：18,216条（51名儿童）',
    '  IEMOCAP：8,525条（仅10名成人）',
    '  → 儿童SER不能直接套用成人经验',
], top=1.2); footer(s, pn)

# ===== SECTION 2 =====
s = ns(); pn += 1; section_slide(s, 2, '核心贡献', '四项可复现的工程规律'); footer(s, pn)

s = ns(); pn += 1; title_bar(s, '四项核心贡献')
body(s, [
    '① 首个儿童SER跨语料库系统比较框架',
    '  3数据集 × 3池化 × 3 seeds × 7系列 = 63组对照实验',
    '  统一流水线 + 说话人独立70/15/15（MD5哈希）+ 零泄露验证',
    '',
    '② FD—WA严格单调关系',
    '  Fréchet Distance定量测量WavLM隐空间中的语料库分布距离',
    '  FD每增加 → WA必下降（增强/年龄/风格三维度，无一例外）',
    '  可在训练前预测跨语料迁移的性能上限',
    '',
    '③ 韵律先验的人群依赖性：儿童中性 vs 成人有害',
    '  C-BESD: Δ(SA−PG)=−0.13pp(n.s.) | FAU: Δ=+1.31pp(n.s.)',
    '  IEMOCAP: Δ=+9.73pp(大幅有害) → 方向反转 = 人群特异性',
    '',
    '④ 成人增强不可迁移 + WavLM中层(L7-L9)一致偏好',
    '  成人数据混入儿童训练 → −30.6pp（损害>随机噪声−21.4pp）',
    '  层融合贡献最大单模块增益+10pp，L7-L9在所有数据集上一致获得最高权重',
], top=1.2, size=13); footer(s, pn)

# ===== SECTION 3 =====
s = ns(); pn += 1; section_slide(s, 3, '数据集与预处理', '三个语料库 · 统一流水线'); footer(s, pn)

s = ns(); pn += 1; title_bar(s, '数据集总览')
img(s, os.path.join(FIG,'v4_dataset_bars.png'), 0.3, 1.0, 7.5, 2.5)
table_slide(s,
    ['数据集', '类别', '样本数', '说话人', '风格', '语言'],
    [['C-BESD', '6类', '4,179', '70名儿童', '演绎式', '英语(L2)'],
     ['FAU Aibo', '4类*', '18,216', '51名儿童 (10-13岁)', '自发性', '德语'],
     ['IEMOCAP', '4类', '8,525', '10名成人', '演绎式', '英语']],
    top=3.7)
body(s, [
    '*FAU 五类→四类重映射（2026-06-10）：A+E→Angry(5,093)  P→Happy(889)  N→Neutral(10,967)  R→Sad(1,267)',
    '关键操作：C-BESD儿童ID从文件名正则提取 → 70名真实儿童（旧方案237个伪ID → 数据泄露风险）',
    '说话人划分：MD5哈希 70/15/15, seed=42, 零重叠验证 | 预处理：16kHz mono, 峰值归一化, 4秒截断, 在线WavLM',
], top=5.5, size=9); footer(s, pn)

# ===== SECTION 4 =====
s = ns(); pn += 1; section_slide(s, 4, '模型架构', 'WavLM + 层融合 + 池化 + SEMLP'); footer(s, pn)

s = ns(); pn += 1; title_bar(s, '模型架构总览')
img(s, os.path.join(FIG,'v4_architecture.png'), 0.3, 1.0, 12.5, 4.0)
body(s, ['WavLM Base：冻结94M参数，94,000h预训练语音 → 768维帧级特征@50Hz',
    '层融合：12个可学习参数，加权求和 → 训练后观测到L7-L9最高权重（非人为指定）',
    '池化：三种策略参数严格匹配(各~111K) → 唯一变量：是否注入F0+能量先验',
    '分类器：SE-MLP ~593K → 总可训~704K (<WavLM骨干1%)'], top=5.2, size=10); footer(s, pn)

s = ns(); pn += 1; title_bar(s, '三种池化策略 — 参数严格匹配的并行对比')
img(s, os.path.join(FIG,'v4_pooling_comparison.png'), 0.3, 1.0, 12.5, 4.5)
body(s, ['核心设计原则：三种池化的可训参数数量严格相同（各~111,105）',
    '唯一变量：是否将逐帧F0(基频)和RMS能量注入注意力计算',
    '→ Self-Attention与Prosody-Guided之间的所有性能差异，都归因于韵律先验的有无',
    '→ Mean Pooling作为无参数基线，量化时序建模本身的价值'], top=5.6, size=10); footer(s, pn)

# ===== SECTION 5 =====
s = ns(); pn += 1; section_slide(s, 5, '实验设计', '7个系列 · 63组对照实验'); footer(s, pn)

s = ns(); pn += 1; title_bar(s, '7个实验系列 — 63组对照实验总览')
img(s, os.path.join(FIG,'v4_experiment_matrix.png'), 0.3, 1.0, 12.5, 4.5)
body(s, ['共计：63组实验 | 3 seeds × 42组 + 1 seed × 21组 | 预估GPU：~46h (RTX 4090D)',
    '固定协议：WavLM冻结 + 层融合 + SEMLP | 说话人独立MD5哈希70/15/15 | 零泄露验证通过',
    '产出：9张发表级三线表 + 10张科研图表 | B1运行中（已完成E1-01~03 C-BESD三池化）'], top=5.6, size=10); footer(s, pn)

# ===== SECTION 6 =====
s = ns(); pn += 1; section_slide(s, 6, '主实验结果', '域内基线 · 零样本矩阵 · 消融'); footer(s, pn)

s = ns(); pn += 1; title_bar(s, 'E1: 域内池化基线 — 三数据集对比')
img(s, os.path.join(FIG,'v4_main_results.png'), 0.3, 1.0, 9.5, 3.8)
body(s, ['★ Self-Attention在三个数据集上均为最优域内池化',
    '★ 韵律先验人群依赖性：儿童(SA≈PG) vs 成人(SA≫PG) — 方向反转',
    '★ C-BESD/FAU: 3-seed mean±std | IEMOCAP: 2-seed (s456 val=0)'],
    top=5.0, size=10); footer(s, pn)

s = ns(); pn += 1; title_bar(s, 'FD—Accuracy 关系详解')
img(s, os.path.join(FIG,'v4_fd_accuracy.png'), 0.2, 1.0, 6.5, 3.8)
body(s, ['FD = Fréchet Distance：WavLM隐空间中两个语料库的分布距离',
    '• 每语料500条 → 768维特征 → 假设高斯分布 → 计算FD',
    '• FD=0为同分布 | FD越大=两个语料库差异越大',
    '',
    'FD作为诊断工具的作用：',
    '• FD每增加 → WA必下降（Spearman ρ≈−0.95）',
    '• 训练前即可预测跨语料性能上限',
    '• 指导语料库选择和增强策略设计',
    '',
    'FAU效果不如IEMOCAP的原因：',
    '• FD只测语料库级距离，不测类内方差',
    '• FAU自发性→类内方差极大（同一情绪表达多样）',
    '• IEMOCAP演绎式→类内方差小（标准化表达）'],
    top=1.0, left=7.0, width=5.5, size=10); footer(s, pn)

s = ns(); pn += 1; title_bar(s, 'E3: 零样本迁移矩阵')
img(s, os.path.join(FIG,'v4_zeroshot_matrix.png'), 0.5, 1.0, 5.5, 4.2)
body(s, ['热力图：对角线=域内训练 | 非对角线=零样本迁移',
    '全部6个跨语料方向<34%，无一可用',
    '（但均>25%随机基线→WavLM有一定跨域鲁棒性）',
    '',
    '关键发现：',
    '• FD(C→F)=8.50 → WA=19.56%',
    '  分布偏移是零样本崩塌的主导因素',
    '• 最高跨语料：IEMOCAP→C-BESD(33.67%)',
    '• 成人→儿童方向略好于反向',
    '• 零样本矩阵本身就是FD-WA单调性的证据'],
    top=1.0, left=6.3, width=5.5, size=10); footer(s, pn)

s = ns(); pn += 1; title_bar(s, '韵律先验的人群依赖性 — 方向反转')
table_slide(s,
    ['数据集','人群','Δ(SA−PG)','显著性','解释'],
    [['C-BESD','儿童演绎式','−0.13pp','不显著(n.s.)','韵律先验中性'],
     ['FAU Aibo','儿童自发性','+1.31pp','不显著(n.s.)','自发场景微弱正向'],
     ['IEMOCAP','成人演绎式','+9.73pp','大幅显著','韵律先验有害！']])
body(s, [
    '原因：成人韵律模式规则 → WavLM中层已编码 → F0+能量变为冗余干扰',
    '儿童F0高(250-330Hz)、变异大 → 韵律与SSL特征互补而非冗余',
    '含义：对儿童SER，Prosody-Guided是安全的；对成人SER则应避免',
    'APC验证：APC_wav=0.74, APC_delta=−0.045（负值=降低了冗余相关）',
], top=2.8, size=11); footer(s, pn)

# ===== SECTION 7 =====
s = ns(); pn += 1; section_slide(s, 7, '核心发现', '四条工程规律 · 四条证据链'); footer(s, pn)

s = ns(); pn += 1; title_bar(s, '发现1 & 3：FD单调性 + 成人增强不可迁移')
table_slide(s,
    ['增强条件 (C-BESD, v5协议)', 'FD', 'Test WA (%)', 'ΔWA'],
    [['C1: 干净训练（基线）','0.00','81.24 ± 0.71','—'],
     ['C2: +成人数据 (IEMOCAP)','9.87','50.61 ± 0.68','−30.63'],
     ['C3: +AWGN (儿童匹配噪声)','8.71','59.89 ± 1.43','−21.35'],
     ['C4: 极端增强 (以上全部)','11.99','46.50 ± 2.31','−34.74']], top=1.2)
body(s, ['发现1：FD每增加→WA必下降。三维度无一例外。Spearman ρ≈−0.95。FD是可靠的跨语料性能预测量。',
    '发现3：成人数据(C2)损害(−30.6pp) > 随机噪声(C3, −21.4pp) — 成人语料对儿童模型是"毒药"。'],
    top=3.5, size=12); footer(s, pn)

s = ns(); pn += 1; title_bar(s, '发现4：WavLM层融合 — 训练后观测')
body(s, [
    '层融合如何工作（训练阶段）：',
    '  12个可学习标量α₁…α₁₂ → softmax → w₁…w₁₂ → H_fused = Σ w_ℓ · H^(ℓ)',
    '  权重通过反向传播与分类器联合优化 — 它们是训练的一部分',
    '',
    '训练后观测到什么（非人为指定，数据驱动的结果）：',
    '  Argmax: L9 = 0.0912（1-based）| 最小值: L4 = 0.0780',
    '  中层(L7-L9) > 低层(L1-L6) > 高层(L10-L12)',
    '  熵 = 2.484（接近均匀值2.485 — 所有层都有贡献，中层系统性偏高）',
    '',
    '融合方式对比：仅用最后一层 < 最优单层(网格搜索) < 12层加权求和',
    '层融合贡献最大的单模块增益：+10pp（相对于仅用最后一层）',
    '',
    'L7-L9偏好是WavLM表示结构的属性，在C-BESD/FAU/IEMOCAP上一致出现 → 非数据集过拟合。',
], top=1.2, size=13); footer(s, pn)

# ===== SECTION 8 =====
s = ns(); pn += 1; section_slide(s, 8, '总结与展望'); footer(s, pn)

s = ns(); pn += 1; title_bar(s, '总结：从"造更大的模型"到"发现可验证的工程规律"')
body(s, [
    '本研究贡献：',
    '  ① 首个儿童SER跨语料库系统比较框架（63组实验，3数据集，严格说话人独立划分）',
    '  ② FD—WA严格单调关系 — FD可作为跨语料性能退化预测量',
    '  ③ 韵律先验人群依赖性 — 儿童中性(Δ≈0) vs 成人有害(Δ=+9.73pp) → 方向反转',
    '  ④ 成人增强不可迁移 + WavLM中层偏好 — 年龄感知的数据策略 + 层融合设计原则',
    '',
    '当前进展：',
    '  ✅ 数据管道修复 ✅ 论文初稿 v2 ✅ PPT v4 ✅ Nature-Skills安装',
    '  🔄 B1 AutoDL运行中（E1-01~03已完成，batch=16，预计~20h完成全部27 runs）',
    '  ⏳ 全63组执行 → 数据填入 → Nature-Polishing终稿 → 投稿',
    '',
    '根本信息：儿童语音情绪识别不是"成人SER + F0调高"',
    '— 它需要分布匹配的数据策略、人群特定的声学先验、和可解释的模型设计。',
], top=1.2, size=12); footer(s, pn)

# ===== FINAL =====
s = ns(); pn += 1
s.background.fill.solid(); s.background.fill.fore_color.rgb = GREEN_D
t1 = s.shapes.add_textbox(Inches(3), Inches(2.5), Inches(7), Inches(1.2))
tf = t1.text_frame; p = tf.paragraphs[0]; p.text = '谢谢！'; p.font.size = Pt(44); p.font.bold = True; p.font.color.rgb = WHITE; p.alignment = PP_ALIGN.CENTER
t2 = s.shapes.add_textbox(Inches(3), Inches(4.0), Inches(7), Inches(0.6))
tf2 = t2.text_frame; p2 = tf2.paragraphs[0]; p2.text = '欢迎提问与讨论'; p2.font.size = Pt(20); p2.font.color.rgb = RGBColor(0xCC,0xDD,0xCC); p2.alignment = PP_ALIGN.CENTER
footer(s, pn)

prs.save(OUT)
print(f'PPT v4 saved: {OUT}')
print(f'Slides: {len(prs.slides)} | Size: {os.path.getsize(OUT)/1024/1024:.1f} MB')
