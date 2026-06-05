const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "许梓超";
pres.title = "分布驱动的儿童语音情绪识别";

const C = {
  dark:"1E293B", slate:"475569", light:"F1F5F9", white:"FFFFFF",
  teal:"0D9488", tealL:"14B8A6", green:"2E8B57", greenL:"6DBF8A",
  orange:"D9793A", blue:"4A90C4", red:"E74C3C", gold:"F1C40F",
  gray:"94A3B8", mute:"CBD5E1",
};

const mkShadow = () => ({ type:"outer", blur:4, offset:2, angle:135, color:"000000", opacity:0.08 });

function titleBar(slide, title, subtitle) {
  slide.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:10, h:1.05, fill:{color:C.dark} });
  slide.addText(title, { x:0.6, y:0.1, w:8.8, h:0.55, fontSize:24, color:C.white, bold:true, margin:0 });
  if(subtitle) slide.addText(subtitle, { x:0.6, y:0.6, w:8.8, h:0.35, fontSize:11, color:C.gray, italic:true, margin:0 });
  slide.addShape(pres.shapes.RECTANGLE, { x:0, y:1.05, w:10, h:0.04, fill:{color:C.teal} });
}
function addSN(slide, n) {
  slide.addText(`${n}`, { x:9.3, y:5.2, w:0.5, h:0.3, fontSize:8, color:C.gray, align:"right" });
}
function statBox(slide, x, y, w, h, number, label, color) {
  slide.addShape(pres.shapes.RECTANGLE, { x,y,w,h, fill:{color:C.white}, shadow:mkShadow() });
  slide.addShape(pres.shapes.RECTANGLE, { x,y,w:0.06,h, fill:{color} });
  slide.addText(number, { x:x+0.2, y:y+0.15, w:w-0.3, h:h*0.5, fontSize:26, color, bold:true, margin:0 });
  slide.addText(label, { x:x+0.2, y:y+h*0.52, w:w-0.3, h:h*0.45, fontSize:9, color:C.slate, margin:0 });
}
function figPH(slide, x, y, w, h, label) {
  slide.addShape(pres.shapes.RECTANGLE, { x,y,w,h, fill:{color:C.light}, line:{color:C.mute,width:1,dashType:"dash"} });
  slide.addText(`[ ${label} ]`, { x,y,w,h, fontSize:10, color:C.gray, align:"center", valign:"middle" });
}

// ═══════════ SLIDE 1: 封面 ═══════════
let s1 = pres.addSlide();
s1.background = { color:C.dark };
s1.addShape(pres.shapes.RECTANGLE, { x:0, y:3.0, w:10, h:0.05, fill:{color:C.teal} });
s1.addText("分布驱动的儿童语音情绪识别", {
  x:0.8, y:1.2, w:8.4, h:1.2, fontSize:32, color:C.white, bold:true, align:"left"
});
s1.addText("Distribution-Driven Child Speech Emotion Recognition\n从儿童语音的统计分布出发，重新发现 SER 的工程规律", {
  x:0.8, y:2.5, w:8.4, h:0.6, fontSize:14, color:C.gray, italic:true
});
s1.addText("许梓超  |  2026年6月", { x:0.8, y:3.5, w:8.4, h:0.4, fontSize:13, color:C.white });
s1.addText("AC Suite 2026-05  ·  WavLM Base (wavlm-base-sv)  ·  C-BESD / FAU Aibo / IEMOCAP", {
  x:0.8, y:4.1, w:8.4, h:0.3, fontSize:10, color:C.gray
});

// ═══════════ SLIDE 2: 目录 ═══════════
let s2 = pres.addSlide();
s2.background = { color:C.white };
titleBar(s2, "目录", "Agenda");
const agenda = [
  ["01","研究背景与动机","儿童SER的核心挑战——分布偏移的发现"],
  ["02","框架架构总览","Pipeline: Waveform → WavLM → LayerFusion → Pooling → SEMLP"],
  ["03","核心方法演进","从DrseNet到SEMLP · 从Mean Pooling到Prosody-Guided Pooling"],
  ["04","实验设计","6组实验 × 3数据集 × 2种池化策略对比"],
  ["05","消融与分析","Pooling对比、FD诊断、儿童特异性、XAI"],
  ["06","总结与展望","从\"通用模型\"到\"儿童SER规律\"的认知转变"],
];
agenda.forEach((a,i) => {
  const y = 1.35 + i*0.65;
  s2.addShape(pres.shapes.OVAL, { x:0.8, y:y+0.05, w:0.4, h:0.4, fill:{color:C.teal} });
  s2.addText(a[0], { x:0.8, y:y+0.05, w:0.4, h:0.4, fontSize:14, color:C.white, bold:true, align:"center", valign:"middle", margin:0 });
  s2.addText(a[1], { x:1.4, y, w:7, h:0.3, fontSize:16, color:C.dark, bold:true, margin:0 });
  s2.addText(a[2], { x:1.4, y:y+0.3, w:7, h:0.25, fontSize:11, color:C.slate, margin:0 });
});
addSN(s2,2);

// ═══════════ SLIDE 3: 问题与动机 ═══════════
let s3 = pres.addSlide();
s3.background = { color:C.white };
titleBar(s3, "研究动机：儿童语音≠成人语音", "主流SSL模型在成人语料预训练 → 儿童语音存在系统性分布偏移");

s3.addText([
  { text:"核心发现", options:{bold:true,breakLine:true,fontSize:15,color:C.dark} },
  { text:"所有主流SSL模型（WavLM, HuBERT, wav2vec 2.0）均在成人语料上预训练", options:{bullet:true,breakLine:true,fontSize:13,color:C.slate} },
  { text:"儿童F0更高（250–400 Hz vs. 80–200 Hz）、共振峰更分散、韵律变异性更大 [Dmitrieva et al., 2008]", options:{bullet:true,breakLine:true,fontSize:13,color:C.slate} },
  { text:"成人SER标配的数据增强直接应用于儿童语音 → 引入分布偏移、损害性能", options:{bullet:true,breakLine:true,fontSize:13,color:C.slate} },
  { text:"现有工作直接复用成人SER管线，缺乏针对儿童语音特性的系统性理解", options:{bullet:true,fontSize:13,color:C.slate} },
], { x:0.6, y:1.3, w:5.2, h:3.5, valign:"top" });

statBox(s3, 6.2, 1.5, 3.2, 0.85, "92.78%", "C-BESD (儿童演绎)\n本文方法", C.green);
statBox(s3, 6.2, 2.55, 3.2, 0.85, "58.67%", "IEMOCAP (成人)\n相同Pipeline", C.blue);
statBox(s3, 6.2, 3.6, 3.2, 0.85, "19.56%", "C-BESD→FAU 零样本\n跨域迁移失败", C.red);
s3.addText("FD(C-BESD, IEMOCAP) = 7.20  |  FD(C-BESD, FAU) = 8.50  |  Fréchet Distance 量化分布偏移", {
  x:0.6, y:4.9, w:9, h:0.3, fontSize:10, color:C.gray, italic:true
});
addSN(s3,3);

// ═══════════ SLIDE 4: 核心思路 ═══════════
let s4 = pres.addSlide();
s4.background = { color:C.white };
titleBar(s4, "核心思路：从\"通用模型\"到\"儿童SER规律\"", "围绕儿童-成人分布偏移，发现三项可验证的工程规律");

const pillars = [
  ["规律1","分布偏移定量诊断","Fréchet Distance 在WavLM隐空间\n量化儿童-成人偏移 (FD=7.20)\nFD↑ ⇒ Accuracy↓ (严格单调负相关)\n→ 偏移是可度量、可预测的", C.teal],
  ["规律2","韵律池化的儿童特异性","F0 + RMS作为注意力先验\n儿童: +2.24pp / 成人: -2.12pp\n→ 韵律先验不是通用增益,\n而是儿童语音特有的信号", C.green],
  ["规律3","中层表征承载关键信息","WavLM 12层融合替代单层输出\nL8权值最高(0.091), L11-12最低\n→ 儿童韵律信息集中在\n中层声学编码层, 非深层语义层", C.orange],
];
pillars.forEach((p,i) => {
  const x = 0.4 + i*3.2;
  s4.addShape(pres.shapes.RECTANGLE, { x, y:1.4, w:2.9, h:3.5, fill:{color:C.white}, shadow:mkShadow() });
  s4.addShape(pres.shapes.RECTANGLE, { x, y:1.4, w:2.9, h:0.06, fill:{color:p[4]} });
  s4.addText(p[0], { x:x+0.2, y:1.6, w:2.5, h:0.25, fontSize:10, color:p[4], bold:true, margin:0 });
  s4.addText(p[1], { x:x+0.2, y:1.9, w:2.5, h:0.5, fontSize:15, color:C.dark, bold:true, margin:0 });
  s4.addText(p[2], { x:x+0.2, y:2.5, w:2.5, h:2.2, fontSize:10.5, color:C.slate, margin:0 });
});
addSN(s4,4);

// ═══════════ SLIDE 5: 架构总览 ═══════════
let s5 = pres.addSlide();
s5.background = { color:C.white };
titleBar(s5, "框架架构总览", "端到端Pipeline: Waveform → WavLM → LayerFusion → Pooling → SEMLP → Output");
figPH(s5, 0.4, 1.3, 9.2, 3.4, "插入 fig00: 系统架构图\n(fig00_architecture_v2.png)");
s5.addText("总可训练参数: ~704K（不到WavLM骨干94.6M的1%）| 冻结骨干 + 仅训练LayerFusion(12权重) + Pooling(111K) + SEMLP(593K)", {
  x:0.6, y:4.78, w:8.8, h:0.35, fontSize:11, color:C.teal, bold:true
});
addSN(s5,5);

// ═══════════ SLIDE 6: WavLM + LayerFusion ═══════════
let s6 = pres.addSlide();
s6.background = { color:C.white };
titleBar(s6, "WavLM Base Backbone + LayerFusion", "冻结SSL编码器 → 12层可学习加权融合 → (B, T, 768)");

s6.addText([
  { text:"WavLM Base (wavlm-base-sv)", options:{bold:true,breakLine:true,fontSize:14,color:C.dark} },
  { text:"94.6M参数, 12层Transformer, 94,000小时成人语音预训练 [Chen et al., IEEE JSTSP 2022]", options:{bullet:true,breakLine:true,fontSize:11.5,color:C.slate} },
  { text:"全部参数冻结——梯度不流入骨干网络", options:{bullet:true,breakLine:true,fontSize:11.5,color:C.slate} },
  { text:"输出13个hidden states: 1个Input Embedding + 12层Transformer输出", options:{bullet:true,breakLine:true,fontSize:11.5,color:C.slate} },
  { text:"每层: (T=200帧, D=768) @ 50Hz帧率 (帧移320样本=20ms)", options:{bullet:true,breakLine:true,fontSize:11.5,color:C.slate} },
  { text:"", options:{breakLine:true,fontSize:5} },
  { text:"WavLM LayerFusion — 替代传统\"仅取最后一层\"", options:{bold:true,breakLine:true,fontSize:14,color:C.dark} },
  { text:"F = Σ wᵢ · Hᵢ (i=1..12, 丢弃Input Embedding), wᵢ = softmax(θᵢ)", options:{bullet:true,breakLine:true,fontSize:11.5,color:C.teal,italic:true} },
  { text:"仅12个可训练标量, 初始化为均匀权重 (1/12)", options:{bullet:true,breakLine:true,fontSize:11.5,color:C.slate} },
  { text:"训练后: Layer 8 权值最高=0.091 (中层韵律层), Layer 10-11 权值最低 (~0.06)", options:{bullet:true,breakLine:true,fontSize:11.5,color:C.slate} },
  { text:"结论: 仅取最深层(L11-12)恰好丢失儿童语音最关键的韵律信息", options:{bullet:true,fontSize:11.5,color:C.red} },
], { x:0.6, y:1.25, w:5.3, h:3.9, valign:"top" });
figPH(s6, 6.2, 1.4, 3.4, 3.4, "插入 fig06:\n12层权重分布\n(fig06_layer_fusion_weights.png)");
addSN(s6,6);

// ═══════════ SLIDE 7: 池化策略对比 ═══════════
let s7 = pres.addSlide();
s7.background = { color:C.white };
titleBar(s7, "时序池化: 两种策略并行对比", "参数量严格对齐 (各111,105) —— 唯一差异: 是否注入F0+RMS韵律先验");

s7.addText("我们同时使用两种池化策略, 不是为了\"选最优\", 而是通过对比来验证韵律先验的作用:", {
  x:0.5, y:1.2, w:9, h:0.35, fontSize:12, color:C.dark, italic:true
});

// Self-Attention
s7.addShape(pres.shapes.RECTANGLE, { x:0.4, y:1.65, w:4.4, h:3.35, fill:{color:C.white}, shadow:mkShadow() });
s7.addShape(pres.shapes.RECTANGLE, { x:0.4, y:1.65, w:4.4, h:0.06, fill:{color:C.blue} });
s7.addText("Self-Attention Pooling（基线/对照）", { x:0.6, y:1.8, w:4, h:0.35, fontSize:14, color:C.blue, bold:true, margin:0 });
s7.addText(
  "a_t = MLP(f_t)   [768→116→100→100→1]\nα_t = softmax(a_t)\nz = Σ α_t · f_t   →  (B, 768)\n\n◆ 纯数据驱动注意力 —— 不注入任何外部先验\n◆ 消融实验的公平基线 (等参数量)\n◆ 验证假设: 如果Self-Attn已足够强,\n  说明WavLM特征本身已编码韵律信息\n\nExp1: 92.78% WA (C-BESD)",
  { x:0.6, y:2.25, w:4, h:2.5, fontSize:10.5, color:C.slate, margin:0 }
);
// Prosody-Guided
s7.addShape(pres.shapes.RECTANGLE, { x:5.2, y:1.65, w:4.4, h:3.35, fill:{color:C.white}, shadow:mkShadow() });
s7.addShape(pres.shapes.RECTANGLE, { x:5.2, y:1.65, w:4.4, h:0.06, fill:{color:C.orange} });
s7.addText("Prosody Guided Pooling（本文方法）", { x:5.4, y:1.8, w:4, h:0.35, fontSize:14, color:C.orange, bold:true, margin:0 });
s7.addText(
  "p_t = MLP_pros([f₀; e_t])   [2→64→64]\nc_t = [f_t; p_t]               832-dim\na_t = MLP_fusion(c_t)    [832→128→1]\nα_t = softmax(a_t),  z = Σ α_t · f_t\n\n◆ F0 (YIN, C2–C7) + RMS Energy 显式先验\n◆ 韵律仅调制注意力权重, 不直接参与分类\n◆ 先验角色: \"特征\" → \"帧选择器\"\n\nExp2: 91.30% WA (C-BESD)",
  { x:5.4, y:2.25, w:4, h:2.5, fontSize:10.5, color:C.slate, margin:0 }
);
addSN(s7,7);

// ═══════════ SLIDE 8: 分类器演进 DrseNet→SEMLP ═══════════
let s8 = pres.addSlide();
s8.background = { color:C.white };
titleBar(s8, "分类器演进: DrseNet → SEMLP", "从复杂CNN到轻量SE-MLP —— 功能聚焦、参数精简");

// Left: DrseNet (old)
s8.addShape(pres.shapes.RECTANGLE, { x:0.4, y:1.3, w:4.4, h:3.2, fill:{color:C.white}, shadow:mkShadow() });
s8.addShape(pres.shapes.RECTANGLE, { x:0.4, y:1.3, w:4.4, h:0.06, fill:{color:C.gray} });
s8.addText("旧版: DrseNet (ResNet+SE+多Stage)", { x:0.6, y:1.45, w:4, h:0.35, fontSize:14, color:C.gray, bold:true, margin:0 });
s8.addText([
  { text:"结构: 多阶段残差卷积 + SE注意力 + 多尺度特征融合", options:{bullet:true,breakLine:true,fontSize:11,color:C.slate} },
  { text:"定位: 项目早期 (v5_622) 的核心分类网络", options:{bullet:true,breakLine:true,fontSize:11,color:C.slate} },
  { text:"问题: 模型复杂, 参数量大, 与冻结SSL骨干的轻量理念不一致", options:{bullet:true,breakLine:true,fontSize:11,color:C.slate} },
  { text:"本质: 在162-dim手工特征时代设计的分类器, 迁移到768-dim SSL特征后存在结构冗余", options:{bullet:true,fontSize:11,color:C.slate} },
], { x:0.6, y:1.9, w:4, h:2.4, valign:"top" });

// Right: SEMLP (new)
s8.addShape(pres.shapes.RECTANGLE, { x:5.2, y:1.3, w:4.4, h:3.2, fill:{color:C.white}, shadow:mkShadow() });
s8.addShape(pres.shapes.RECTANGLE, { x:5.2, y:1.3, w:4.4, h:0.06, fill:{color:C.teal} });
s8.addText("新版: SEMLP (SE-MLP, ~593K)", { x:5.4, y:1.45, w:4, h:0.35, fontSize:14, color:C.teal, bold:true, margin:0 });
s8.addText([
  { text:"结构: 768→512→SE-Block→256→128→4", options:{bullet:true,breakLine:true,fontSize:11,color:C.slate} },
  { text:"SEBlock(h) = h ⊙ σ(W₂·ReLU(W₁·h)), W₁:512→32, W₂:32→512", options:{bullet:true,breakLine:true,fontSize:11,color:C.teal,italic:true} },
  { text:"功能: 自适应通道门控——增强判别维度, 抑制噪声维度", options:{bullet:true,breakLine:true,fontSize:11,color:C.slate} },
  { text:"理念: Pooling输出的已是单一(768,)全局向量, 无需复杂时序建模", options:{bullet:true,breakLine:true,fontSize:11,color:C.slate} },
  { text:"效果: ~593K参数, 配合冻结骨干, 总可训参数<704K——不到骨干1%", options:{bullet:true,fontSize:11,color:C.slate} },
], { x:5.4, y:1.9, w:4, h:2.4, valign:"top" });

// Bottom: transition narrative
s8.addText([
  { text:"演进逻辑: ", options:{bold:true,fontSize:12,color:C.dark} },
  { text:"当特征从162-dim手工特征升级为768-dim SSL特征, 且时序建模职责前移到Pooling层后, 分类器的角色从\"特征提取+分类\"简化为\"纯分类\"。SEMLP以1/10的参数量完成了DrseNet的工作, 同时SE-Block保留了通道级自适应校准能力。", options:{fontSize:11,color:C.slate} },
], { x:0.5, y:4.6, w:9, h:0.6, valign:"top" });
addSN(s8,8);

// ═══════════ SLIDE 9: 为什么F0+RMS ═══════════
let s9 = pres.addSlide();
s9.background = { color:C.white };
titleBar(s9, "为什么选择 F0 + RMS Energy？", "基于儿童情绪声学研究证据的韵律特征选择");

s9.addText([
  { text:"实证依据（儿童特异性研究）", options:{bold:true,breakLine:true,fontSize:14,color:C.dark} },
  { text:"Dmitrieva et al. (2008): F0 和 F1 是7–17岁儿童情绪韵律感知最重要的声学参数", options:{bullet:true,breakLine:true,fontSize:12,color:C.slate} },
  { text:"Hubbard (1991): F0均值/范围 + 振幅变化率是最具判别力的婴幼儿情绪特征 (9–13个月)", options:{bullet:true,breakLine:true,fontSize:12,color:C.slate} },
  { text:"Kao et al. (2022): F0均值和强度变化是3–12月龄婴儿情绪处理的优先线索", options:{bullet:true,breakLine:true,fontSize:12,color:C.slate} },
  { text:"", options:{breakLine:true,fontSize:5} },
  { text:"计算可行性", options:{bold:true,breakLine:true,fontSize:14,color:C.dark} },
  { text:"YIN算法: O(n log n), 对非周期性儿童语音鲁棒, C2–C7覆盖儿童F0范围 [de Cheveigné & Kawahara, 2002]", options:{bullet:true,breakLine:true,fontSize:12,color:C.slate} },
  { text:"RMS: O(n), 对瞬态噪声鲁棒, SER领域最通用的短时能量度量 [Schröder, 2001]", options:{bullet:true,breakLine:true,fontSize:12,color:C.slate} },
  { text:"两者均从波形直接提取——无需ASR、无需强制对齐", options:{bullet:true,breakLine:true,fontSize:12,color:C.slate} },
  { text:"", options:{breakLine:true,fontSize:5} },
  { text:"情绪空间互补覆盖", options:{bold:true,breakLine:true,fontSize:14,color:C.dark} },
  { text:"F0 → 效价 + 语调轮廓  |  RMS → 唤醒度  |  两者共同覆盖valence-arousal模型双轴 [Juslin & Laukka, 2003]", options:{bullet:true,breakLine:true,fontSize:12,color:C.slate} },
  { text:"仅2维特征 (vs. eGeMAPS的88维) 保持韵律分支轻量性, 确保总可训参数 < 704K", options:{bullet:true,fontSize:12,color:C.slate} },
], { x:0.6, y:1.25, w:8.8, h:3.9, valign:"top" });
addSN(s9,9);

// ═══════════ SLIDE 10: 数据集 ═══════════
let s10 = pres.addSlide();
s10.background = { color:C.white };
titleBar(s10, "数据集概览", "三语料库——覆盖年龄、语言和自发性维度");

const datasets = [
  ["C-BESD","马来语","儿童\n(演绎式)","2,780","angry, happy,\nneutral, sad","4,179→2,780\n(6→4类映射)",C.green],
  ["FAU Aibo","德语","儿童\n(自发性)","18,216","angry, happy,\nneutral, sad","51名儿童(10-13岁)\nSony Aibo机器人交互",C.orange],
  ["IEMOCAP","英语","成人\n(演绎+即兴)","8,525","angry, happy,\nneutral, sad","10名演员, 5组双人会话\n~12小时, 多模态",C.blue],
];
datasets.forEach((d,i) => {
  const y = 1.3 + i*1.3;
  s10.addShape(pres.shapes.RECTANGLE, { x:0.4, y, w:9.2, h:1.1, fill:{color:C.white}, shadow:mkShadow() });
  s10.addShape(pres.shapes.RECTANGLE, { x:0.4, y, w:0.06, h:1.1, fill:{color:d[6]} });
  s10.addText(d[0], { x:0.7, y:y+0.08, w:1.4, h:0.3, fontSize:15, color:d[6], bold:true, margin:0 });
  s10.addText(d[1], { x:0.7, y:y+0.38, w:1.4, h:0.25, fontSize:10, color:C.gray, margin:0 });
  s10.addText(d[2], { x:2.2, y:y+0.15, w:1.1, h:0.7, fontSize:11, color:C.slate, margin:0 });
  s10.addText(d[3], { x:3.3, y:y+0.2, w:0.9, h:0.35, fontSize:16, color:C.dark, bold:true, margin:0 });
  s10.addText("条样本", { x:3.3, y:y+0.55, w:0.9, h:0.2, fontSize:9, color:C.gray, margin:0 });
  s10.addText(d[4], { x:4.3, y:y+0.15, w:1.6, h:0.7, fontSize:11, color:C.slate, margin:0 });
  s10.addText(d[5], { x:6.1, y:y+0.15, w:3.3, h:0.7, fontSize:10, color:C.gray, italic:true, margin:0 });
});
addSN(s10,10);

// ═══════════ SLIDE 11: 实验协议 ═══════════
let s11 = pres.addSlide();
s11.background = { color:C.white };
titleBar(s11, "实验协议", "AC Suite 2026-05: 6组实验, 2种池化, 3个数据集");

s11.addText([
  { text:"数据划分与预处理", options:{bold:true,breakLine:true,fontSize:14,color:C.dark} },
  { text:"说话人独立MD5哈希划分: 70% train / 15% val / 15% test (说话人零重叠)", options:{bullet:true,breakLine:true,fontSize:12,color:C.slate} },
  { text:"统一4类情绪空间: {angry, happy, neutral, sad}; C-BESD 6→4映射 (丢弃DISGUST, FEAR)", options:{bullet:true,breakLine:true,fontSize:12,color:C.slate} },
  { text:"音频: 16kHz mono, peak normalization, 4s截断 (200帧 @ 50Hz)", options:{bullet:true,breakLine:true,fontSize:12,color:C.slate} },
  { text:"优化器: AdamW (lr=3e-4), CosineAnnealing (Tmax=100), Early Stop (patience=15)", options:{bullet:true,breakLine:true,fontSize:12,color:C.slate} },
  { text:"损失函数: Label-Smoothing Cross-Entropy (ε=0.1 default, ε=0.15 FAU)", options:{bullet:true,breakLine:true,fontSize:12,color:C.slate} },
  { text:"", options:{breakLine:true,fontSize:5} },
  { text:"两套正则化配置", options:{bold:true,breakLine:true,fontSize:14,color:C.dark} },
  { text:"Default (Exp1–4): wd=1e-3, ls=0.1, 无dropout, 无grad_clip —— 用于演绎式语音", options:{bullet:true,breakLine:true,fontSize:12,color:C.slate} },
  { text:"FAU (Exp5/5b): wd=5e-3, ls=0.15, pooling_dropout=0.3, grad_clip=1.0 —— 防止小样本自发性数据快速过拟合", options:{bullet:true,fontSize:12,color:C.slate} },
], { x:0.6, y:1.25, w:8.8, h:4.0, valign:"top" });
addSN(s11,11);

// ═══════════ SLIDE 12: 6组实验总览 ═══════════
let s12 = pres.addSlide();
s12.background = { color:C.white };
titleBar(s12, "6组实验设计", "覆盖4维度：池化对比、年龄特异性、风格泛化、跨数据集一致性验证");

const exps = [
  ["Exp1","C-BESD","C-BESD","Self-Attn","default","92.78%","儿童演绎式天花板",C.green],
  ["Exp2","C-BESD","C-BESD","Prosody","default","91.30%","核心消融：韵律先验",C.greenL],
  ["Exp3","IEMOCAP","IEMOCAP","Prosody","default","58.67%","成人对照",C.blue],
  ["Exp4","C-BESD","FAU","Prosody","default","19.56%","零样本跨域迁移",C.red],
  ["Exp5","FAU","FAU","Prosody","fau","66.36%","自发性域内训练",C.orange],
  ["Exp5b","FAU","FAU","Self-Attn","fau","66.18%","FAU场景消融验证",C.orange],
];
const headerY = 1.25;
const colW = [0.65,0.95,0.95,1.05,0.75,0.95,2.95];
const cols = ["实验","训练集","测试集","Pooling","正则","WA","实验目的"];
let hx = 0.3;
cols.forEach((c,i) => {
  s12.addText(c, { x:hx, y:headerY, w:colW[i], h:0.3, fontSize:10, color:C.white, bold:true, align:"center", margin:0 });
  hx += colW[i];
});
s12.addShape(pres.shapes.RECTANGLE, { x:0.3, y:headerY, w:9.2, h:0.3, fill:{color:C.dark} });
exps.forEach((e,i) => {
  let x = 0.3;
  const y = headerY + 0.35 + i*0.52;
  const bg = i%2===0 ? C.white : C.light;
  s12.addShape(pres.shapes.RECTANGLE, { x:0.3, y, w:9.2, h:0.47, fill:{color:bg} });
  s12.addShape(pres.shapes.RECTANGLE, { x:0.3, y, w:0.04, h:0.47, fill:{color:e[7]} });
  colW.forEach((cw,j) => {
    const isWA = j===5;
    s12.addText(e[j], { x, y, w:cw, h:0.47, fontSize:isWA?12:9, color:isWA?C.dark:C.slate, bold:isWA, align:"center", valign:"middle", margin:0 });
    x += cw;
  });
});
addSN(s12,12);

// ═══════════ SLIDE 13: 主结果 ═══════════
let s13 = pres.addSlide();
s13.background = { color:C.white };
titleBar(s13, "主实验结果", "6组实验 WA + UAR 柱状图");
figPH(s13, 0.3, 1.25, 9.4, 3.3, "插入 fig01: 主实验结果柱状图\n(fig01_main_matrix_wa_uar.png)");
s13.addText([
  { text:"关键发现", options:{bold:true,breakLine:true,fontSize:11,color:C.dark} },
  { text:"① Self-Attn略优于Prosody (92.78% vs 91.30%, Δ=-1.48pp) —— WavLM特征已足够强, 纯注意力即可捕获判别帧", options:{bullet:true,breakLine:true,fontSize:10,color:C.slate} },
  { text:"② 儿童→成人骤降: 91.30%→58.67% (-32.63pp) —— 验证了分布偏移的灾难性影响", options:{bullet:true,breakLine:true,fontSize:10,color:C.slate} },
  { text:"③ 零样本跨域失败: C-BESD→FAU仅19.56% (<随机25%) —— 演绎→自发分布鸿沟不可直接跨越", options:{bullet:true,breakLine:true,fontSize:10,color:C.slate} },
  { text:"④ 域内训练可部分恢复: +46.80pp (19.56%→66.36%) —— 但仍有~26pp天花板差距", options:{bullet:true,fontSize:10,color:C.slate} },
], { x:0.6, y:4.6, w:8.8, h:0.7, valign:"top" });
addSN(s13,13);

// ═══════════ SLIDE 14: 消融合并 — C-BESD + FAU + 混淆矩阵 ═══════════
let s14 = pres.addSlide();
s14.background = { color:C.white };
titleBar(s14, "消融实验: 池化策略对比 + 混淆矩阵", "C-BESD与FAU Aibo两个数据集上的一致结论");

// Left: C-BESD
s14.addShape(pres.shapes.RECTANGLE, { x:0.2, y:1.15, w:4.7, h:2.0, fill:{color:C.white}, shadow:mkShadow() });
s14.addShape(pres.shapes.RECTANGLE, { x:0.2, y:1.15, w:4.7, h:0.04, fill:{color:C.green} });
s14.addText("C-BESD (儿童演绎·马来语)", { x:0.35, y:1.22, w:4.4, h:0.25, fontSize:11, color:C.green, bold:true, margin:0 });
figPH(s14, 0.35, 1.5, 2.1, 1.5, "插入 fig05:\nC-BESD池化对比");
figPH(s14, 2.55, 1.5, 2.2, 1.5, "插入 fig07+08:\n混淆矩阵 Exp1+Exp2");

// Right: FAU
s14.addShape(pres.shapes.RECTANGLE, { x:5.1, y:1.15, w:4.7, h:2.0, fill:{color:C.white}, shadow:mkShadow() });
s14.addShape(pres.shapes.RECTANGLE, { x:5.1, y:1.15, w:4.7, h:0.04, fill:{color:C.orange} });
s14.addText("FAU Aibo (儿童自发性·德语)", { x:5.25, y:1.22, w:4.4, h:0.25, fontSize:11, color:C.orange, bold:true, margin:0 });
figPH(s14, 5.25, 1.5, 4.4, 1.5, "插入 fig02:\nFAU池化对比 + 误差线");

// Bottom row: IEMOCAP, Zero-shot, FAU confusion matrices
s14.addText("更多混淆矩阵 (Exp3–Exp5b):", { x:0.3, y:3.25, w:4, h:0.25, fontSize:11, color:C.dark, bold:true, margin:0 });
figPH(s14, 0.2, 3.5, 1.8, 1.5, "插入 figA1:\nExp3 IEMOCAP");
figPH(s14, 2.15, 3.5, 1.8, 1.5, "插入 figA2:\nExp4 Zero-Shot FAU");
figPH(s14, 4.1, 3.5, 1.8, 1.5, "插入 figA3:\nExp5 FAU Prosody");
figPH(s14, 6.05, 3.5, 1.8, 1.5, "插入 figA4:\nExp5b FAU Self-Attn");

// Bottom takeaway
s14.addText([
  { text:"一致结论: 两种池化策略性能相当——Self-Attn已足够强, 韵律先验提供了额外的可解释性维度(APC_wav=0.718)而非性能增益。跨数据集混淆矩阵确认了儿童→成人、演绎→自发的性能梯度。", options:{fontSize:10,color:C.slate} },
], { x:0.3, y:5.05, w:9.2, h:0.25, valign:"top" });
addSN(s14,14);

// ═══════════ SLIDE 15: FD vs Accuracy ═══════════
let s15 = pres.addSlide();
s15.background = { color:C.white };
titleBar(s15, "分布偏移诊断: FD vs Accuracy", "Fréchet Distance定量预测分类性能退化——严格单调负相关");

figPH(s15, 0.3, 1.2, 5.5, 3.9, "插入 fig03:\nFD vs Accuracy 散点图\n(fig03_fd_vs_accuracy.png)");

s15.addText([
  { text:"FD-Accuracy 对应关系", options:{bold:true,breakLine:true,fontSize:13,color:C.dark} },
  { text:"", options:{breakLine:true,fontSize:4} },
  { text:"FD=0.00 → 92.78% (C-BESD域内)", options:{breakLine:true,fontSize:11,color:C.green,bold:true} },
  { text:"同语料基线——无分布偏移，性能最佳", options:{breakLine:true,fontSize:10,color:C.gray} },
  { text:"", options:{breakLine:true,fontSize:4} },
  { text:"FD=7.20 → 58.67% (IEMOCAP成人)", options:{breakLine:true,fontSize:11,color:C.blue,bold:true} },
  { text:"年龄偏移——儿童→成人性能骤降 -32.63pp", options:{breakLine:true,fontSize:10,color:C.gray} },
  { text:"", options:{breakLine:true,fontSize:4} },
  { text:"FD=8.50 → 66.36% (FAU域内)", options:{breakLine:true,fontSize:11,color:C.orange,bold:true} },
  { text:"风格+自发性偏移——域内训练可部分克服", options:{breakLine:true,fontSize:10,color:C.gray} },
  { text:"", options:{breakLine:true,fontSize:4} },
  { text:"FD=8.50 → 19.56% (FAU零样本)", options:{breakLine:true,fontSize:11,color:C.red,bold:true} },
  { text:"相同FD, WA差异 = +46.80pp → 域内训练是最有效的偏移补偿", options:{breakLine:true,fontSize:10,color:C.gray} },
  { text:"", options:{breakLine:true,fontSize:4} },
  { text:"核心规律: FD ↑ ⇒ Accuracy ↓", options:{fontSize:12,color:C.dark,bold:true} },
  { text:"FD在AutoDL GPU上重算 (2026-05-27, max_samples=500, seed=42)", options:{fontSize:10,color:C.gray,italic:true} },
], { x:6.1, y:1.25, w:3.6, h:4.0, valign:"top" });
addSN(s15,15);

// ═══════════ SLIDE 16: 儿童特异性 ═══════════
let s16 = pres.addSlide();
s16.background = { color:C.white };
titleBar(s16, "儿童特异性验证: Prosody Δ 跨数据集梯度", "韵律池化增益随数据集\"儿童相关性\"单调递减");

s16.addShape(pres.shapes.RECTANGLE, { x:0.5, y:1.3, w:9, h:3.6, fill:{color:C.white}, shadow:mkShadow() });
const ds2 = [
  { name:"C-BESD\n(儿童演绎)", a1:78.61, a3:80.85, delta:"+2.24pp", dc:C.green, icon:"↑" },
  { name:"CREMA-D\n(成人演绎)", a1:64.69, a3:65.44, delta:"+0.75pp", dc:C.gray, icon:"→" },
  { name:"IEMOCAP\n(成人演绎)", a1:54.50, a3:52.38, delta:"-2.12pp", dc:C.red, icon:"↓" },
];
ds2.forEach((d,i) => {
  const x = 0.8 + i*3;
  s16.addText(d.name, { x, y:1.5, w:2.5, h:0.55, fontSize:13, color:C.dark, bold:true, align:"center", margin:0 });
  s16.addText(`A1 (mean pool): ${d.a1}%`, { x, y:2.15, w:2.5, h:0.25, fontSize:11, color:C.slate, align:"center", margin:0 });
  s16.addText(`A3 (prosody): ${d.a3}%`, { x, y:2.45, w:2.5, h:0.25, fontSize:11, color:C.slate, align:"center", margin:0 });
  s16.addText(`Δ = ${d.delta}`, { x, y:2.85, w:2.5, h:0.5, fontSize:20, color:d.dc, bold:true, align:"center", margin:0 });
  s16.addText(d.icon, { x, y:3.45, w:2.5, h:0.35, fontSize:24, color:d.dc, align:"center", margin:0 });
});
s16.addText(
  "规律: 韵律先验增益随\"儿童相关性\"单调递减 — +2.24 → +0.75 → -2.12 pp。韵律引导的时序池化具有明确的儿童特异性，而非通用SER增益模块。",
  { x:0.6, y:4.3, w:8.8, h:0.5, fontSize:11, color:C.slate, italic:true }
);
addSN(s16,16);

// ═══════════ SLIDE 17: XAI ═══════════
let s17 = pres.addSlide();
s17.background = { color:C.white };
titleBar(s17, "XAI可解释性: 注意力-韵律显著图", "Prosody Guided Pooling 如何聚焦情绪显著帧");

figPH(s17, 0.3, 1.2, 9.4, 3.2, "插入 fig04: XAI 三联显著图\n(使用 canonical 版本:\npaper_draft/figures/fig04_xai_saliency_triple.png\n或 results/xai_final.png)");

s17.addText([
  { text:"APC_wav = 0.718  |  APC_delta = -0.144 (canonical data, 2026-05-27)", options:{bold:true,breakLine:true,fontSize:13,color:C.dark} },
  { text:"APC_wav (Attention-RMS correlation): 强正相关——注意力峰值与能量爆发区域对齐", options:{bullet:true,breakLine:true,fontSize:11,color:C.slate} },
  { text:"APC_delta < 0: 注意力抑制了原始音高追踪——模型学到了超越简单韵律相关的情绪判别模式", options:{bullet:true,breakLine:true,fontSize:11,color:C.slate} },
  { text:"红色saliency区域集中在高能量、情绪表达强烈的语音段——静音帧被正确忽略", options:{bullet:true,breakLine:true,fontSize:11,color:C.slate} },
  { text:"⚠ 使用 canonical 版本 (APC=0.718), 非6月2日旧版 (APC=0.515已废弃)", options:{bullet:true,fontSize:11,color:C.red} },
], { x:0.6, y:4.5, w:8.8, h:0.75, valign:"top" });
addSN(s17,17);

// ═══════════ SLIDE 18: 总结 ═══════════
let s18 = pres.addSlide();
s18.background = { color:C.white };
titleBar(s18, "总结: 从\"通用模型\"到\"儿童SER规律\"", "我们最初的目标是设计一个通用的儿童语音情绪识别模型，但实验过程让我们发现了更本质的东西");

s18.addText([
  { text:"研究的认知转变", options:{bold:true,breakLine:true,fontSize:15,color:C.dark} },
  { text:"", options:{breakLine:true,fontSize:5} },
  { text:"起点——工程目标: 设计一个在儿童语音上\"好用\"的SER模型，替换掉成人SER的直接迁移方案", options:{bullet:true,breakLine:true,fontSize:13,color:C.slate} },
  { text:"过程——发现规律: ", options:{bullet:true,breakLine:true,fontSize:13,color:C.slate} },
  { text:"规律① 分布偏移是可定量预测的: FD与Accuracy严格单调负相关, 偏移越大→性能越低", options:{bullet:true,breakLine:true,fontSize:12,color:C.teal} },
  { text:"规律② 儿童语音的关键信息在中层而非深层: WavLM L8权值最高, L11-12权值最低——传统\"取最后一层\"恰好丢失了最关键的信息", options:{bullet:true,breakLine:true,fontSize:12,color:C.teal} },
  { text:"规律③ F0+RMS韵律先验具有儿童特异性, 不是通用增益: +2.24pp(儿童)→ +0.75pp(成人CREMA-D)→ -2.12pp(成人IEMOCAP)", options:{bullet:true,breakLine:true,fontSize:12,color:C.teal} },
  { text:"规律④ 成人SER的增强经验不可迁移: 所有增强操作均引入分布偏移并降低性能", options:{bullet:true,breakLine:true,fontSize:12,color:C.teal} },
  { text:"", options:{breakLine:true,fontSize:5} },
  { text:"终点——认知贡献: 这项工作的价值不在于\"提出了一个新模型\"，而在于通过系统性实验揭示了儿童SER中几条可验证、可复现的工程规律。模型的92.78%只是这些规律的自然结果。", options:{fontSize:13,color:C.dark,bold:true} },
], { x:0.5, y:1.3, w:9, h:3.9, valign:"top" });
addSN(s18,18);

// ═══════════ SLIDE 19: 后续计划 ═══════════
let s19 = pres.addSlide();
s19.background = { color:C.white };
titleBar(s19, "后续计划", "待办事项与研究方向");

s19.addText([
  { text:"近期任务", options:{bold:true,breakLine:true,fontSize:15,color:C.dark} },
  { text:"完成全部图表 (fig00–fig08 + figA1–A4) 的最终生成——使用 AC Suite canonical 数据", options:{bullet:true,breakLine:true,fontSize:13,color:C.slate} },
  { text:"完成论文初稿——22篇真实参考文献已收集并验证", options:{bullet:true,breakLine:true,fontSize:13,color:C.slate} },
  { text:"在全量C-BESD测试集上重算APC (当前仅1条样本)", options:{bullet:true,breakLine:true,fontSize:13,color:C.slate} },
  { text:"", options:{breakLine:true,fontSize:5} },
  { text:"研究拓展", options:{bold:true,breakLine:true,fontSize:15,color:C.dark} },
  { text:"Domain Adaptation: 弥合演绎式→自发性差距 (66% vs 93%天花板)", options:{bullet:true,breakLine:true,fontSize:13,color:C.slate} },
  { text:"多语言儿童SER规律验证: 利用BESD English + Telugu子集检验发现的规律是否跨语言成立", options:{bullet:true,breakLine:true,fontSize:13,color:C.slate} },
  { text:"探索更多韵律特征的儿童特异性: jitter, shimmer, HNR在guided pooling框架下的Δ梯度", options:{bullet:true,breakLine:true,fontSize:13,color:C.slate} },
  { text:"投稿目标: Interspeech / ICASSP / IEEE TAC", options:{bullet:true,fontSize:13,color:C.slate} },
], { x:0.6, y:1.35, w:8.8, h:3.7, valign:"top" });
addSN(s19,19);

// ═══════════ SLIDE 20: 致谢 ═══════════
let s20 = pres.addSlide();
s20.background = { color:C.dark };
s20.addShape(pres.shapes.RECTANGLE, { x:0, y:2.2, w:10, h:0.05, fill:{color:C.teal} });
s20.addText("感谢聆听", { x:0.5, y:0.9, w:9, h:1.0, fontSize:44, color:C.white, bold:true, align:"center", margin:0 });
s20.addText("Questions & Discussion", { x:0.5, y:2.4, w:9, h:0.8, fontSize:20, color:C.gray, align:"center", margin:0 });
s20.addText([
  { text:"许梓超", options:{breakLine:true,fontSize:14,color:C.white} },
  { text:"分布驱动的儿童语音情绪识别", options:{breakLine:true,fontSize:12,color:C.gray} },
  { text:"AC Suite 2026-05  ·  WavLM Base (wavlm-base-sv)", options:{fontSize:11,color:C.gray} },
], { x:0.5, y:3.4, w:9, h:1.2, align:"center" });
addSN(s20,20);

pres.writeFile({ fileName:"C:/Users/59892/Desktop/儿童SER_项目汇报.pptx" })
  .then(() => console.log("DONE: 儿童SER_项目汇报.pptx"))
  .catch(err => console.error("ERROR:", err));
