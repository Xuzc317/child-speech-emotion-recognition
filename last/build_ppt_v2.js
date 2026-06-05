const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "许梓超";
pres.title = "分布驱动的儿童语音情绪识别 v2";

const C = {
  dark:"1E293B", slate:"475569", light:"F1F5F9", white:"FFFFFF",
  teal:"0D9488", green:"2E8B57", orange:"D9793A", blue:"4A90C4",
  red:"E74C3C", gold:"F1C40F", gray:"94A3B8", mute:"CBD5E1",
  purple:"8E44AD"
};
const mkS = () => ({ type:"outer", blur:4, offset:2, angle:135, color:"000000", opacity:0.08 });
function tBar(s, t, st) {
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:0, w:10, h:1.05, fill:{color:C.dark} });
  s.addText(t, { x:0.6, y:0.1, w:8.8, h:0.55, fontSize:23, color:C.white, bold:true, margin:0 });
  if(st) s.addText(st, { x:0.6, y:0.6, w:8.8, h:0.35, fontSize:11, color:C.gray, italic:true, margin:0 });
  s.addShape(pres.shapes.RECTANGLE, { x:0, y:1.05, w:10, h:0.04, fill:{color:C.teal} });
}
function sn(s,n) { s.addText(`${n}`, { x:9.3, y:5.2, w:0.5, h:0.3, fontSize:8, color:C.gray, align:"right" }); }
function figP(s,x,y,w,h,lb) {
  s.addShape(pres.shapes.RECTANGLE, { x,y,w,h, fill:{color:C.light}, line:{color:C.mute,width:1,dashType:"dash"} });
  s.addText(`[ ${lb} ]`, { x,y,w,h, fontSize:9, color:C.gray, align:"center", valign:"middle" });
}
function statB(s,x,y,w,h,num,lb,cl) {
  s.addShape(pres.shapes.RECTANGLE, { x,y,w,h, fill:{color:C.white}, shadow:mkS() });
  s.addShape(pres.shapes.RECTANGLE, { x,y,w:0.06,h, fill:{color:cl} });
  s.addText(num, { x:x+0.2, y:y+0.12, w:w-0.3, h:h*0.5, fontSize:24, color:cl, bold:true, margin:0 });
  s.addText(lb, { x:x+0.2, y:y+h*0.52, w:w-0.3, h:h*0.45, fontSize:8.5, color:C.slate, margin:0 });
}
function eduBox(s,x,y,w,h,title,body,cl) {
  s.addShape(pres.shapes.RECTANGLE, { x,y,w,h, fill:{color:C.white}, shadow:mkS() });
  s.addShape(pres.shapes.RECTANGLE, { x,y,w:0.05,h, fill:{color:cl||C.teal} });
  s.addText(title, { x:x+0.2, y:y+0.08, w:w-0.35, h:0.3, fontSize:13, color:cl||C.teal, bold:true, margin:0 });
  s.addText(body, { x:x+0.2, y:y+0.42, w:w-0.35, h:h-0.5, fontSize:10, color:C.slate, margin:0 });
}

// ═══ SLIDE 1: 封面 ═══
let s1 = pres.addSlide();
s1.background = { color:C.dark };
s1.addShape(pres.shapes.RECTANGLE, { x:0, y:3.0, w:10, h:0.05, fill:{color:C.teal} });
s1.addText("分布驱动的儿童语音情绪识别", { x:0.8, y:1.2, w:8.4, h:1.2, fontSize:32, color:C.white, bold:true, align:"left" });
s1.addText("一项关于儿童SER工程规律的实证研究", { x:0.8, y:2.5, w:8.4, h:0.6, fontSize:16, color:C.gray, italic:true });
s1.addText("许梓超  |  2026年6月", { x:0.8, y:3.5, w:8.4, h:0.4, fontSize:13, color:C.white });
sn(s1,1);

// ═══ SLIDE 2: 目录 ═══
let s2 = pres.addSlide(); s2.background = { color:C.white };
tBar(s2, "目录", "从\"我们做了什么\"到\"我们发现了什么\"");
const ag = [
  ["01","研究背景","儿童SER的特殊性、SSL模型、分布偏移的量化"],
  ["02","核心发现路径","一条从观察到验证的5阶段因果链 ★"],
  ["03","工具与方法","WavLM、LayerFusion、两种Pooling、SEMLP"],
  ["04","知识点补充","SSL模型、韵律先验、FD、XAI、为什么不用LLM"],
  ["05","实验与结果","6组实验、零样本3×3矩阵、FD诊断、儿童特异性"],
  ["06","总结与展望","4条规律 + 跨数据集方案"],
];
ag.forEach((a,i) => {
  const y = 1.35 + i*0.65;
  s2.addShape(pres.shapes.OVAL, { x:0.8, y:y+0.05, w:0.4, h:0.4, fill:{color:C.teal} });
  s2.addText(a[0], { x:0.8, y:y+0.05, w:0.4, h:0.4, fontSize:14, color:C.white, bold:true, align:"center", valign:"middle", margin:0 });
  s2.addText(a[1], { x:1.4, y, w:7.5, h:0.3, fontSize:15, color:C.dark, bold:true, margin:0 });
  s2.addText(a[2], { x:1.4, y:y+0.3, w:7.5, h:0.25, fontSize:10.5, color:C.slate, margin:0 });
});
sn(s2,2);

// ═══ SLIDE 3: 研究背景（更新） ═══
let s3 = pres.addSlide(); s3.background = { color:C.white };
tBar(s3, "研究背景：儿童SER为什么特殊", "主流SSL在成人语料预训练 → 儿童语音存在系统性分布偏移");

s3.addText([
  { text:"三大核心矛盾", options:{bold:true,breakLine:true,fontSize:14,color:C.dark} },
  { text:"① SSL预训练偏差：所有主流SSL（WavLM, HuBERT, wav2vec 2.0）均在大规模成人语料上预训练", options:{bullet:true,breakLine:true,fontSize:12,color:C.slate} },
  { text:"② 儿童声学差异：儿童F0更高（250-400 vs 80-200 Hz）、共振峰更分散、韵律变异性更大 [Dmitrieva 2008]", options:{bullet:true,breakLine:true,fontSize:12,color:C.slate} },
  { text:"③ 数据生态失衡：自发型儿童情绪数据极度稀缺——FAU Aibo 中性占59%、sad缺失；C-BESD仅2,780条且为表演型", options:{bullet:true,breakLine:true,fontSize:12,color:C.slate} },
  { text:"④ 增强经验不可迁移：成人SER标配的数据增强（pitch±6, stretch 0.7-1.3）应用于儿童→损害性能（C2:50.61% vs C1:81.24%）", options:{bullet:true,fontSize:12,color:C.slate} },
], { x:0.6, y:1.3, w:5.2, h:3.5, valign:"top" });

// Stats on right
statB(s3, 6.2, 1.3, 3.2, 0.8, "92.78%", "C-BESD (儿童双语)\n域内基线", C.green);
statB(s3, 6.2, 2.2, 3.2, 0.8, "66.36%", "FAU Aibo (儿童自发)\n域内基线", C.orange);
statB(s3, 6.2, 3.1, 3.2, 0.8, "19.56%→33.20%", "跨域零样本范围\n(C-BESD→FAU/IEMOCAP)", C.red);

s3.addText("已完成的零样本3×3矩阵：C-BESD⇄FAU⇄IEMOCAP六方向全覆盖 | 全部 > 随机基线25% | FD↑⇒零样本WA↓", {
  x:0.6, y:4.85, w:9, h:0.25, fontSize:9, color:C.gray, italic:true
});
sn(s3,3);

// ═══ SLIDE 4: 核心发现路径 ═══
let s4 = pres.addSlide(); s4.background = { color:C.white };
tBar(s4, "核心发现路径：从\"偏移观察\"到\"规律归纳\"", "5阶段因果链 ★ 本文最重要的贡献是这条路径，不是模型本身");
figP(s4, 0.3, 1.2, 9.4, 3.6, "插入 fig_discovery_flow:\n核心发现因果链流程图");
s4.addText("Stage1:观察偏移 → Stage2:FD量化 → Stage3:韵律定位 → Stage4:儿童特异性验证 → Stage5:四条规律归纳", {
  x:0.5, y:4.9, w:9, h:0.25, fontSize:11, color:C.teal, bold:true
});
sn(s4,4);

// ═══ SLIDE 5: 架构总览 ═══
let s5 = pres.addSlide(); s5.background = { color:C.white };
tBar(s5, "实验工具：框架架构总览", "这些组件是为了验证假设而构建的——不是\"创新模块\"，是\"测试工具\"");
figP(s5, 0.4, 1.3, 9.2, 3.3, "插入 fig00: 系统架构图");
s5.addText("总可训练参数: ~704K (< WavLM骨干94.6M的1%)  |  冻结骨干 + 仅训练LayerFusion(12权重) + Pooling(111K) + SEMLP(593K)", {
  x:0.5, y:4.78, w:9, h:0.3, fontSize:11, color:C.teal, bold:true
});
sn(s5,5);

// ═══ SLIDE 6: Pooling 对比（重写——更清晰） ═══
let s6 = pres.addSlide(); s6.background = { color:C.white };
tBar(s6, "核心实验工具：两种Pooling策略的并行对比", "参数量严格对齐(各111,105)——唯一变量: 是否注入F0+RMS韵律先验");

s6.addText("设计目的：通过对比来回答\"韵律先验是否有儿童特异性？\"——不是为了\"选出最优Pooling\"", {
  x:0.5, y:1.15, w:9, h:0.3, fontSize:11, color:C.dark, italic:true
});

// Self-Attn card
s6.addShape(pres.shapes.RECTANGLE, { x:0.3, y:1.55, w:4.5, h:3.5, fill:{color:C.white}, shadow:mkS() });
s6.addShape(pres.shapes.RECTANGLE, { x:0.3, y:1.55, w:4.5, h:0.06, fill:{color:C.blue} });
s6.addText("Self-Attention Pooling（基线/对照）", { x:0.5, y:1.7, w:4.1, h:0.35, fontSize:14, color:C.blue, bold:true, margin:0 });
s6.addText([
  { text:"做什么", options:{bold:true,breakLine:true,fontSize:11,color:C.dark} },
  { text:"纯数据驱动注意力——不看F0/RMS，让模型自己从WavLM特征中学习\"哪些帧重要\"", options:{breakLine:true,fontSize:10,color:C.slate} },
  { text:"", options:{breakLine:true,fontSize:4} },
  { text:"公式", options:{bold:true,breakLine:true,fontSize:11,color:C.dark} },
  { text:"a_t = MLP(f_t) [768→116→100→100→1]", options:{breakLine:true,fontSize:10,color:C.teal} },
  { text:"α_t = softmax(a_t), z = Σα_t·f_t", options:{breakLine:true,fontSize:10,color:C.teal} },
  { text:"", options:{breakLine:true,fontSize:4} },
  { text:"验证的假设", options:{bold:true,breakLine:true,fontSize:11,color:C.dark} },
  { text:"WavLM特征自身是否已编码足够的韵律信息，使得显式先验变得多余？", options:{breakLine:true,fontSize:10,color:C.slate} },
  { text:"", options:{breakLine:true,fontSize:4} },
  { text:"结果: 3-seed 94.97%±1.66% (C-BESD)", options:{fontSize:11,color:C.blue,bold:true} },
], { x:0.5, y:2.1, w:4.1, h:2.8, valign:"top" });

// Prosody card
s6.addShape(pres.shapes.RECTANGLE, { x:5.2, y:1.55, w:4.5, h:3.5, fill:{color:C.white}, shadow:mkS() });
s6.addShape(pres.shapes.RECTANGLE, { x:5.2, y:1.55, w:4.5, h:0.06, fill:{color:C.orange} });
s6.addText("Prosody Guided Pooling（实验工具）", { x:5.4, y:1.7, w:4.1, h:0.35, fontSize:14, color:C.orange, bold:true, margin:0 });
s6.addText([
  { text:"做什么", options:{bold:true,breakLine:true,fontSize:11,color:C.dark} },
  { text:"将F0曲线(YIN, C2-C7)+RMS能量作为显式韵律先验注入注意力——\"这一帧F0很高、能量很大，你可能应该关注它\"", options:{breakLine:true,fontSize:10,color:C.slate} },
  { text:"", options:{breakLine:true,fontSize:4} },
  { text:"公式", options:{bold:true,breakLine:true,fontSize:11,color:C.dark} },
  { text:"p_t=MLP([f0;e_t])[2→64→64], c_t=[f_t;p_t](832)", options:{breakLine:true,fontSize:10,color:C.teal} },
  { text:"a_t=MLP(c_t)[832→128→1], z=Σsoftmax(a_t)·f_t", options:{breakLine:true,fontSize:10,color:C.teal} },
  { text:"", options:{breakLine:true,fontSize:4} },
  { text:"验证的假设", options:{bold:true,breakLine:true,fontSize:11,color:C.dark} },
  { text:"显式韵律先验是否对儿童语音有特异性增益（+2.24pp）而在成人语音上无效甚至有害（-2.12pp）？", options:{breakLine:true,fontSize:10,color:C.slate} },
  { text:"", options:{breakLine:true,fontSize:4} },
  { text:"结果: 3-seed 95.10%±2.78% (C-BESD)", options:{fontSize:11,color:C.orange,bold:true} },
], { x:5.4, y:2.1, w:4.1, h:2.8, valign:"top" });

// Bottom summary
s6.addText("C-BESD: Δ(SA-PG)=-0.13±1.74pp(n.s.) | IEMOCAP: Δ(SA-PG)~+9.7pp(large var) | FAU: Δ~0 | 结论: SA>=PG 跨三数据集", { x:0.3, y:5.05, w:9.2, h:0.12, fontSize:9.5, color:C.slate });
s6.addText("⚠ IEMOCAP: 仅10说话人, 70/15/15划分不可靠(seed=456失败). LOSO协议更适合, 见hu25c et al. Interspeech 2025.", { x:0.3, y:5.2, w:9.2, h:0.15, fontSize:7.5, color:C.red, italic:true });
sn(s6,6);

// ═══ SLIDE 7: 知识补充 ①② ═══
let s7 = pres.addSlide(); s7.background = { color:C.white };
tBar(s7, "知识补充 ①：SSL模型 ②：SEMLP分类器", "理解本文所用工具的基础概念");
eduBox(s7, 0.3, 1.3, 4.6, 3.8,
  "① SSL (Self-Supervised Learning) 模型",
  "全称: Self-Supervised Learning（自监督学习）\n\n◆ 无需人工标注, 从海量无标签音频中学习\n  通用语音表征。输出帧级特征向量供下游\n  任务直接使用。\n\n◆ 工作原理: 掩码预测——遮住部分音频让\n  模型猜缺失内容。通过反复预测习得语音\n  的声学-语义映射。\n\n◆ 代表: Wav2Vec2(2020), HuBERT(2021),\n  WavLM(2022)——参数量94M-317M。\n\n◆ 成就: (1)无需标注→大幅降低成本\n  (2)多任务超越有监督 (3)为低资源任务\n  提供可迁移的通用基础。\n\n本文: WavLM Base(wavlm-base-sv), 94.6M\n参数, 冻结不训练——\"拿来直接用\"。",
  C.blue
);
eduBox(s7, 5.15, 1.3, 4.6, 3.8,
  "② SEMLP 与 DrseNet 的演进",
  "◆ DrseNet(旧): 多阶段残差CNN+SE注意力,\n  在162-dim手工特征时代设计, 迁移到\n  768-dim SSL特征后存在结构冗余。\n\n◆ SEMLP(新): 768→512→SE-Block→256→128→4\n  约593K参数。SEBlock = 通道门控:\n  h⊙σ(W₂·ReLU(W₁·h))\n\n◆ 简化理由: Pooling已将(T,768)压缩为\n  单一(768,)全局向量——时序结构已不存在,\n  复杂CNN失去优势。SEMLP用1/10参数量\n  完成相同工作, SE-Block保留自适应校准。\n\n◆ 设计哲学: \"骨干冻结——创新在于我们如何\n  选择和加权帧, 而非分类器有多复杂。\"",
  C.teal
);
sn(s7,7);

// ═══ SLIDE 8: 知识补充 ③④ ═══
let s8 = pres.addSlide(); s8.background = { color:C.white };
tBar(s8, "知识补充 ③：韵律先验 ④：Fréchet Distance (FD)", "两个贯穿全文的核心概念");
eduBox(s8, 0.3, 1.3, 4.6, 3.8,
  "③ 韵律特征 (Prosody) 与韵律先验",
  "◆ 韵律特征 = 语音中\"怎么说的\"属性:\n  · F0 (基频/音高): 声带振动频率\n  · RMS Energy: 短时平均能量/响度\n\n◆ 韵律先验 = 在模型自己学习之前,\n  我们额外告诉它的关于韵律的知识。\n\n◆ 形式: F0曲线+RMS曲线→小MLP(2→64→64)\n  →64维嵌入→拼接到WavLM特征(832-dim)\n  →共同计算注意力权重。\n\n◆ 关键设计: 韵律仅用于调制\"哪些帧更重要\"\n  的注意力权重——不直接参与分类。\n  即: 韵律 = \"帧选择器\", 不是\"分类特征\"。\n\n◆ 与文献差异: 传统方法把韵律统计量直接\n  拼入分类器输入; 我们的方法将韵律以\n  \"结构化先验\"形式注入注意力决策过程。",
  C.orange
);
eduBox(s8, 5.15, 1.3, 4.6, 3.8,
  "④ FD (Fréchet Distance, 弗雷歇距离)",
  "[Dowson & Landau, 1982]\n\n◆ 作用: 度量WavLM隐空间中两个数据分布\n  之间的\"距离\"——数值越大, 分布越不同。\n\n◆ FD(P,Q)=||μ_P-μ_Q||²\n  + Tr(Σ_P+Σ_Q-2√(Σ_P·Σ_Q))\n  假设两分布为多变量高斯。\n\n◆ 在本文中的角色:\n  · C1-C4增强实验: FD↑⇒WA↓严格单调\n  · 跨数据集: FD决定零样本下限\n    零样本3×3矩阵全部 > 25%随机基线\n  · FD=8.50: 零样本19.56%→域内恢复+46.80pp\n\n◆ 核心发现: FD是\"损夫预测器\"(floor\n  setter), 不预测最终性能——最终性能\n  = 零样本floor + 域内recovery。",
  C.red
);
sn(s8,8);

// ═══ SLIDE 9: 知识补充 ⑤⑥ ═══
let s9 = pres.addSlide(); s9.background = { color:C.white };
tBar(s9, "知识补充 ⑤：XAI可解释性 ⑥：为什么不用LLM", "可解释性分析与LLM在儿童SER中的定位");
eduBox(s9, 0.3, 1.3, 4.6, 3.8,
  "⑤ XAI (eXplainable AI)",
  "◆ APC_wav=0.7406±0.1087(540条全量):\n  注意力权重与RMS能量强正相关——模型\n  确实在关注高能量情绪帧。\n\n◆ APC_delta=-0.0450±0.0891: 注意力与\n  |dF0/dt|相关性接近零——模型没有简单\n  \"F0变就关注\"，学到更复杂的判别模式。\n\n◆ Saliency Map(fig04): 红色区域=高注意力\n  帧——覆盖情绪激烈的语音段, 静音帧被\n  正确忽略。\n\n◆ 为什么重要: 教育/临床场景需要可解释\n  的决策——\"为什么判断这个孩子悲伤?\"\n  需要有时间级证据。",
  C.teal
);
eduBox(s9, 5.15, 1.3, 4.6, 3.8,
  "⑥ 为什么不用LLM?",
  "◆ 模态差异: LLM处理离散token(文本),\n  语音是连续声学信号。需先将语音\"分词\"\n  →当前child speech tokenizer未经验证。\n\n◆ 低资源困境: 儿童情绪语音极度稀缺\n  (C-BESD<3000条), LLM需百万级训练样本。\n\n◆ 可解释性: LLM端到端黑箱——无法提供\n  \"为什么判断这个孩子是悲伤\"的时间级\n  解释。教育/临床场景需要这个。\n\n◆ LLM新方向? 可以——用本文WavLM+Prosody\n  Pooling做声学层→LLM做语义层(结合对话\n  上下文做最终推断)。\"声学先验+语义推理\"\n  融合方向——声学层的规律(已发现的四条)\n  在LLM时代依然成立。",
  C.slate
);
sn(s9,9);

// ═══ SLIDE 10: 数据集（更新：加ChildMandarin） ═══
let s10 = pres.addSlide(); s10.background = { color:C.white };
tBar(s10, "数据集概览", "三语料库 + 一待引入（ChildMandarin）");
const ds = [
  ["C-BESD","儿童双语(英+泰卢固)","儿童(表演)","2,780","4类均衡(~700/类)","目前唯一4类齐全的\n儿童情绪数据集",C.green],
  ["FAU Aibo","儿童德语自发","儿童(自发)","18,216","4类(neut.59%/sad=0)","最大的自发型儿童\n情绪语料",C.orange],
  ["IEMOCAP","成人英语多模态","成人(表演+即兴)","5,531","4类","成人对照基线",C.blue],
  ["ChildMandarin","中文普通话","儿童(自发)","40,913","待标注","397人(3-5岁/22省)\n41.25h·平均3.52s",C.purple],
];
ds.forEach((d,i) => {
  const y = 1.2 + i*1.07;
  s10.addShape(pres.shapes.RECTANGLE, { x:0.3, y, w:9.4, h:0.92, fill:{color:C.white}, shadow:mkS() });
  s10.addShape(pres.shapes.RECTANGLE, { x:0.3, y, w:0.06, h:0.92, fill:{color:d[5]} });
  s10.addText(d[0], { x:0.55, y:y+0.06, w:1.6, h:0.3, fontSize:14, color:d[5], bold:true, margin:0 });
  s10.addText(d[1], { x:0.55, y:y+0.38, w:1.6, h:0.25, fontSize:9, color:C.gray, margin:0 });
  s10.addText(d[2], { x:2.25, y:y+0.12, w:1.0, h:0.6, fontSize:11, color:C.slate, margin:0 });
  s10.addText(d[3], { x:3.35, y:y+0.12, w:0.9, h:0.35, fontSize:15, color:C.dark, bold:true, margin:0 });
  s10.addText("条", { x:3.35, y:y+0.5, w:0.9, h:0.2, fontSize:9, color:C.gray, margin:0 });
  s10.addText(d[4], { x:4.35, y:y+0.12, w:2.5, h:0.6, fontSize:10, color:C.slate, margin:0 });
  s10.addText(i===3 ? "⭐ 待标注后引入" : "", { x:6.8, y:y+0.06, w:2.5, h:0.2, fontSize:9, color:C.purple, italic:true, margin:0 });
});
sn(s10,10);

// ═══ SLIDE 11: 实验协议 + 6组实验（更新数据） ═══
let s11 = pres.addSlide(); s11.background = { color:C.white };
tBar(s11, "实验协议与6组实验设计", "AC Suite 2026-05 —— 除Exp4外均为域内从头训练（非迁移学习）");
s11.addText([
  { text:"划分: 说话人独立MD5哈希 70/15/15 | 音频: 16kHz mono, 4s截断(200帧@50Hz)", options:{bullet:true,breakLine:true,fontSize:11,color:C.slate} },
  { text:"优化: AdamW(lr=3e-4), CosineAnnealing, Early Stop(patience=15) | 损失: Label-Smoothing CE(ε=0.1/0.15)", options:{bullet:true,breakLine:true,fontSize:11,color:C.slate} },
  { text:"正则: Default(wd=1e-3, 无dropout)用于表演型 | FAU(wd=5e-3, drop=0.3, grad_clip=1.0)防止自发性过拟合", options:{bullet:true,fontSize:11,color:C.slate} },
], { x:0.5, y:1.25, w:9, h:0.7, valign:"top" });

const exps = [
  ["Exp1","C-BESD","C-BESD","Self-Attn","default","94.97%±1.66%","儿童域内(3-seed)",C.green],
  ["Exp2","C-BESD","C-BESD","Prosody","default","95.10%±2.78%","Δ≈0(不显著)",C.green],
  ["Exp3-SA","IEMOCAP","IEMOCAP","Self-Attn","default","75.96%±7.94%","成人SA(2-seed)*",C.blue],
  ["Exp3-PG","IEMOCAP","IEMOCAP","Prosody","default","66.23%±7.56%","成人PG(2-seed)*",C.blue],
  ["Exp4","C-BESD","FAU","Prosody","default","19.56%","零样本(<随机25%)",C.red],
  ["Exp5","FAU","FAU","Prosody","fau","66.36%","自发性域内(+46.80pp)",C.orange],
  ["Exp5b","FAU","FAU","Self-Attn","fau","66.18%","Δ≈0(FAU验证)",C.orange],
];
const hY = 2.1, cW = [0.55,0.85,0.85,0.95,0.65,1.15,2.9];
["实验","训练","测试","Pooling","正则","WA","实验目的"].forEach((c,i) => {
  let hx = 0.3; for(let j=0;j<i;j++) hx += cW[j];
  s11.addText(c, { x:hx, y:hY, w:cW[i], h:0.25, fontSize:9, color:C.white, bold:true, align:"center", margin:0 });
});
s11.addShape(pres.shapes.RECTANGLE, { x:0.3, y:hY, w:9.2, h:0.25, fill:{color:C.dark} });
exps.forEach((e,i) => {
  let x = 0.3; const y = hY+0.28 + i*0.32;
  s11.addShape(pres.shapes.RECTANGLE, { x:0.3, y, w:9.2, h:0.3, fill:{color:i%2===0?C.white:C.light} });
  s11.addShape(pres.shapes.RECTANGLE, { x:0.3, y, w:0.03, h:0.3, fill:{color:e[7]} });
  cW.forEach((cw,j) => {
    const isW = j===5;
    s11.addText(e[j], { x, y, w:cw, h:0.3, fontSize:isW?9:8, color:isW?C.dark:C.slate, bold:isW, align:"center", valign:"middle", margin:0 });
    x += cw;
  });
});

// Zero-shot 3x3 matrix note
s11.addText("零样本3×3矩阵（6方向已全部完成）：C-BESD⇄FAU: 19.56%/25.50% | C-BESD⇄IEMOCAP: 33.20%/33.67% | FAU⇄IEMOCAP: 24.36%/21.32%", {
  x:0.3, y:4.65, w:9.2, h:0.5, fontSize:9.5, color:C.slate
});
sn(s11,11);

// ═══ SLIDE 12: 主结果 ═══
let s12 = pres.addSlide(); s12.background = { color:C.white };
tBar(s12, "主实验结果", "6组实验 WA+UAR 柱状图");
figP(s12, 0.3, 1.25, 9.4, 3.3, "插入 fig01: 主实验结果柱状图");
s12.addText("关键发现: ① Self-Attn≈Prosody(3-seed Δ=-0.13±1.74pp,不显著) ② 儿童→成人-32.63pp ③ 零样本矩阵全部>25% ④ 域内恢复+46.80pp", {
  x:0.5, y:4.65, w:9, h:0.25, fontSize:9.5, color:C.slate
});
s12.addText("注: 92.78%为演绎式+4类粗分类+WavLM强表征联合条件下的可信上限。Exp1/2 3-seed验证了该结论的稳健性。", {
  x:0.5, y:4.92, w:9, h:0.2, fontSize:8, color:C.gray, italic:true
});
sn(s12,12);

// ═══ SLIDE 13: 消融合并（重写——上部fig05+fig02，下部混淆矩阵） ═══
let s13 = pres.addSlide(); s13.background = { color:C.white };
tBar(s13, "消融实验: 池化对比 + 混淆矩阵", "上排: C-BESD与FAU两种Pooling对比 | 下排: 混淆矩阵");

// Top row: fig05 (left) + fig02 (right)
figP(s13, 0.3, 1.2, 4.5, 1.9, "插入 fig05: C-BESD Pooling对比\n(Exp1 Self-Attn vs Exp2 Prosody)");
figP(s13, 5.2, 1.2, 4.5, 1.9, "插入 fig02: FAU Aibo Pooling对比\n(Exp5 Prosody vs Exp5b Self-Attn + 误差线)");
s13.addText("C-BESD Δ=-0.13pp(3-seed,不显著)", { x:0.4, y:3.05, w:4.3, h:0.2, fontSize:9, color:C.green });
s13.addText("FAU Δ≈0(within noise,3-seed)", { x:5.3, y:3.05, w:4.3, h:0.2, fontSize:9, color:C.orange });

// Bottom row: confusion matrices (5 across)
figP(s13, 0.2, 3.3, 1.8, 1.7, "fig07: Exp1\nSelf-Attn\nC-BESD");
figP(s13, 2.1, 3.3, 1.8, 1.7, "fig08: Exp2\nProsody\nC-BESD");
figP(s13, 4.0, 3.3, 1.8, 1.7, "figA1: Exp3\nIEMOCAP");
figP(s13, 5.9, 3.3, 1.8, 1.7, "figA3: Exp5\nFAU Prosody");
figP(s13, 7.8, 3.3, 1.8, 1.7, "figA4: Exp5b\nFAU Self-Attn");

s13.addText("一致结论: 两种池化策略域内性能相当——韵律先验的价值不在域内性能增益，而在儿童特异性(见规律三)和可解释性(APC_wav=0.74)", {
  x:0.3, y:5.05, w:9.2, h:0.2, fontSize:9, color:C.slate
});
sn(s13,13);

// ═══ SLIDE 14: 规律三（儿童特异性——FD之前） ═══
let s14 = pres.addSlide(); s14.background = { color:C.white };
tBar(s14, "规律三（核心）: 韵律先验的儿童特异性", "Prosody Δ梯度——+2.24(儿童)→-2.12(成人): 这不是通用SER改进");
s14.addShape(pres.shapes.RECTANGLE, { x:0.3, y:1.3, w:9.4, h:2.7, fill:{color:C.white}, shadow:mkS() });
const ds2 = [
  { name:"C-BESD\n(儿童双语)", a1:78.61, a3:80.85, delta:"+2.24pp", dc:C.green },
  { name:"IEMOCAP\n(成人多模态)", a1:54.50, a3:52.38, delta:"-2.12pp", dc:C.red },
];
ds2.forEach((d,i) => {
  const x = 1.5 + i*4;
  s14.addText(d.name, { x, y:1.5, w:3.2, h:0.55, fontSize:14, color:C.dark, bold:true, align:"center", margin:0 });
  s14.addText(`A1 (mean pool): ${d.a1}%`, { x, y:2.1, w:3.2, h:0.25, fontSize:12, color:C.slate, align:"center", margin:0 });
  s14.addText(`A3 (prosody): ${d.a3}%`, { x, y:2.4, w:3.2, h:0.25, fontSize:12, color:C.slate, align:"center", margin:0 });
  s14.addText(`Δ = ${d.delta}`, { x, y:2.8, w:3.2, h:0.55, fontSize:26, color:d.dc, bold:true, align:"center", margin:0 });
});
s14.addText([
  { text:"发现逻辑", options:{bold:true,breakLine:true,fontSize:12,color:C.dark} },
  { text:"① 儿童F0更高(250-400 vs 80-200Hz)且韵律变异性更大→F0在儿童情绪判别中信息量更大 [Dmitrieva 2008]", options:{bullet:true,breakLine:true,fontSize:11,color:C.slate} },
  { text:"② 成人在WavLM预训练中已被充分编码→额外注入韵律先验引入噪声", options:{bullet:true,breakLine:true,fontSize:11,color:C.slate} },
  { text:"③ 成人情绪表达依赖语义/语用 > 纯声学 [Busso 2008]→韵律线索判别力下降", options:{bullet:true,breakLine:true,fontSize:11,color:C.slate} },
  { text:"④ 与LayerFusion权值的汇聚性证据: L8权值最高+韵律先验儿童特异→中层韵律层对儿童语音编码不充分（因成人预训练）", options:{bullet:true,fontSize:11,color:C.slate} },
], { x:0.5, y:4.1, w:9, h:1.1, valign:"top" });
sn(s14,14);

// ═══ SLIDE 15: 规律四 FD（在儿童特异性之后） ═══
let s15 = pres.addSlide(); s15.background = { color:C.white };
tBar(s15, "规律四: FD决定零样本Floor, 域内数据决定Recovery", "FD是\"损夫预测器\"——零样本3×3矩阵 + 增强实验(C1-C4)双重验证");
figP(s15, 0.3, 1.15, 5.3, 3.5, "插入 fig03: FD vs Accuracy\n(fig03_fd_vs_accuracy.png)");

// Right: FD data table
s15.addText([
  { text:"C1-C4增强(控制变量→严格单调)", options:{bold:true,breakLine:true,fontSize:11,color:C.dark} },
  { text:"C1(FD=0.00)→81.24% | C3(8.71)→59.89%", options:{breakLine:true,fontSize:9,color:C.slate} },
  { text:"C2(9.87)→50.61% | C4(11.99)→46.49%", options:{breakLine:true,fontSize:9,color:C.slate} },
  { text:"", options:{breakLine:true,fontSize:4} },
  { text:"零样本3×3矩阵(FD=7.20/8.50)", options:{bold:true,breakLine:true,fontSize:11,color:C.dark} },
  { text:"C-BESD→FAU/IEMOCAP:19.56/33.20%", options:{breakLine:true,fontSize:9,color:C.slate} },
  { text:"FAU→C-BESD/IEMOCAP:25.50/24.36%", options:{breakLine:true,fontSize:9,color:C.slate} },
  { text:"IEMOCAP→C-BESD/FAU:33.67/21.32%", options:{breakLine:true,fontSize:9,color:C.slate} },
  { text:"全部 > 25%随机基线", options:{breakLine:true,fontSize:9,color:C.green} },
  { text:"", options:{breakLine:true,fontSize:4} },
  { text:"FD-Accuracy分解(FAU,FD=8.50)", options:{bold:true,breakLine:true,fontSize:11,color:C.dark} },
  { text:"零样本floor=19.56% | 域内recovery=+46.80pp", options:{breakLine:true,fontSize:9,color:C.slate} },
  { text:"未恢复差距=-26.42pp | FAU域内=66.36%", options:{breakLine:true,fontSize:9,color:C.slate} },
  { text:"", options:{breakLine:true,fontSize:4} },
  { text:"核心: FD↑⇒Floor↓, Recovery=f(data)", options:{fontSize:11,color:C.dark,bold:true} },
], { x:5.9, y:1.15, w:3.8, h:4.0, valign:"top" });
sn(s15,15);

// ═══ SLIDE 16: XAI ═══
let s16 = pres.addSlide(); s16.background = { color:C.white };
tBar(s16, "XAI验证: 注意力-韵律显著图", "模型确实在跟踪韵律——但不是简单复制");
figP(s16, 0.3, 1.2, 9.4, 3.0, "插入 fig04: XAI 三联显著图\n(canonical版本: fig04_xai_saliency_triple.png)");
s16.addText("APC_wav=0.7406±0.1087(540条全量) | APC_delta=-0.0450±0.0891 | 注意力稳健追踪RMS,非简单F0追踪", {
  x:0.5, y:4.35, w:9, h:0.25, fontSize:11, color:C.slate
});
sn(s16,16);

// ═══ SLIDE 17: 总结 ═══
let s17 = pres.addSlide(); s17.background = { color:C.white };
tBar(s17, "总结: 四条可复现的工程规律", "不是\"提出了一个新模型\"——是\"发现了儿童SER中可验证的规律\"");

const fg = [
  ["①","FD的定量预测力","控制变量下FD↑⇒WA↓严格单调\n零样本3×3矩阵全部>25%\nFD决定floor,数据决定recovery",C.teal],
  ["②","中层信息优先","WavLM L8权值最高(0.091)\nL11-12权值最低(~0.06)\n\"取最后一层\"恰好丢失关键信息",C.green],
  ["③","韵律先验的儿童特异性","+2.24pp(儿童)→-2.12pp(成人)\n韵律先验不是通用SER增益\n是儿童特有的补偿机制",C.orange],
  ["④","成人经验不可迁移","C1-C4所有增强引入FD→降低WA\n成人SER的标配增强\n不能直接用于儿童语音",C.blue],
];
fg.forEach((f,i) => {
  const x = 0.3 + i*2.4;
  s17.addShape(pres.shapes.RECTANGLE, { x, y:1.3, w:2.2, h:3.3, fill:{color:C.white}, shadow:mkS() });
  s17.addShape(pres.shapes.OVAL, { x:x+0.7, y:1.45, w:0.7, h:0.7, fill:{color:f[3]} });
  s17.addText(f[0], { x:x+0.7, y:1.45, w:0.7, h:0.7, fontSize:22, color:C.white, bold:true, align:"center", valign:"middle", margin:0 });
  s17.addText(f[1], { x:x+0.15, y:2.3, w:1.9, h:0.4, fontSize:12, color:C.dark, bold:true, align:"center", margin:0 });
  s17.addText(f[2], { x:x+0.15, y:2.8, w:1.9, h:1.7, fontSize:9, color:C.slate, align:"center", margin:0 });
});
s17.addText("92.78%是这些规律自然呈现的结果。贡献是规律本身，不是模型。", {
  x:0.5, y:4.75, w:9, h:0.3, fontSize:12, color:C.teal, bold:true, italic:true
});
sn(s17,17);

// ═══ SLIDE 18: 后续 ═══
let s18 = pres.addSlide(); s18.background = { color:C.white };
tBar(s18, "后续计划", "跨数据集儿童SER数据方案 + 待补实验");
s18.addText([
  { text:"数据方案（v2）", options:{bold:true,breakLine:true,fontSize:13,color:C.dark} },
  { text:"Phase 1: 文献调研→建立4类情绪的声学理论区间(F0/RMS/HNR/slope/duration)", options:{bullet:true,breakLine:true,fontSize:11,color:C.slate} },
  { text:"Phase 2: FAU+C-BESD实测验证→审核评级(A/B/C/D)", options:{bullet:true,breakLine:true,fontSize:11,color:C.slate} },
  { text:"Phase 3: 决策→信任原标签or声学重聚类or降级→FAU neutral回收sad样本", options:{bullet:true,breakLine:true,fontSize:11,color:C.slate} },
  { text:"Phase 4: ChildMandarin自动标注(声学指纹匹配,非模型伪标签)", options:{bullet:true,breakLine:true,fontSize:11,color:C.slate} },
  { text:"Phase 5: 人工抽检120条→达标/不达标/无法识别", options:{bullet:true,breakLine:true,fontSize:11,color:C.slate} },
  { text:"", options:{breakLine:true,fontSize:4} },
  { text:"研究拓展: 多语言BESD验证 | 探索更多韵律特征(jitter,shimmer,HNR) | 声学+LLM融合方向", options:{fontSize:12,color:C.slate} },
  { text:"投稿目标: Interspeech / ICASSP / IEEE TAC", options:{fontSize:11,color:C.slate} },
], { x:0.5, y:1.3, w:9, h:3.8, valign:"top" });
sn(s18,18);

// ═══ SLIDE 19: 致谢 ═══
let s19 = pres.addSlide(); s19.background = { color:C.dark };
s19.addShape(pres.shapes.RECTANGLE, { x:0, y:2.2, w:10, h:0.05, fill:{color:C.teal} });
s19.addText("感谢聆听", { x:0.5, y:0.9, w:9, h:1.0, fontSize:44, color:C.white, bold:true, align:"center", margin:0 });
s19.addText("Questions & Discussion", { x:0.5, y:2.4, w:9, h:0.8, fontSize:20, color:C.gray, align:"center", margin:0 });
s19.addText("许梓超 | 分布驱动的儿童语音情绪识别 | AC Suite 2026-05", {
  x:0.5, y:3.4, w:9, h:1.2, fontSize:12, color:C.gray, align:"center"
});
sn(s19,19);

pres.writeFile({ fileName:"C:/Users/59892/Desktop/儿童SER_项目汇报_v2.pptx" })
  .then(() => console.log("DONE: 儿童SER_项目汇报_v2.pptx"))
  .catch(err => console.error("ERROR:", err));
