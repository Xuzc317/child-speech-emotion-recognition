"""Build BIT-themed PPTX template with 6 master slide layouts."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu, Cm
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

# ============================================================
# BIT Beamer 设计参数
# ============================================================
PRIMARY = RGBColor(0x02, 0x4A, 0x15)   # 深绿 #024A15
ACCENT  = RGBColor(0x39, 0x82, 0x07)   # 亮绿 #398207
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
DARK    = RGBColor(0x1A, 0x1A, 0x1A)
GRAY    = RGBColor(0x99, 0x99, 0x99)
LGRAY   = RGBColor(0xF0, 0xF0, 0xF0)

W = Inches(13.333)
H = Inches(7.5)

prs = Presentation()
prs.slide_width = W
prs.slide_height = H

# ============================================================
# 工具函数
# ============================================================
def add_rect(slide, left, top, width, height, fill_color=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.line.fill.background()
    if fill_color:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_color
    return shape

def add_tb(slide, left, top, width, height, text, size=18, color=DARK,
           bold=False, align=PP_ALIGN.LEFT, font='Arial'):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.color.rgb = color
    p.font.bold = bold
    p.font.name = font
    p.alignment = align
    return txBox

def add_footer(slide, author='作者 | 单位', title='论文标题', page='N / N'):
    """标准页脚：三栏（作者 | 标题 | 页码）"""
    add_rect(slide, Inches(0), Inches(7.2), W, Pt(0.5), fill_color=GRAY)
    add_tb(slide, Inches(0.5), Inches(7.25), Inches(4), Inches(0.3),
           author, size=9, color=GRAY)
    add_tb(slide, Inches(4.5), Inches(7.25), Inches(4), Inches(0.3),
           title, size=9, color=GRAY, align=PP_ALIGN.CENTER)
    add_tb(slide, Inches(10), Inches(7.25), Inches(3), Inches(0.3),
           page, size=9, color=GRAY, align=PP_ALIGN.RIGHT)

# ============================================================
# Slide 1: 标题页
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_rect(slide, Inches(0), Inches(0), W, Inches(3.2), fill_color=PRIMARY)
add_rect(slide, Inches(0.1), Inches(3.4), W - Inches(0.2), Inches(0.06), fill_color=ACCENT)
add_tb(slide, Inches(0.8), Inches(1.0), Inches(11), Inches(1.5),
       '论文标题 Title Here', size=36, color=WHITE, bold=True)
add_tb(slide, Inches(0.8), Inches(2.3), Inches(11), Inches(0.7),
       '副标题 Subtitle Here', size=20, color=RGBColor(0xCC, 0xCC, 0xCC))
add_tb(slide, Inches(0.8), Inches(4.0), Inches(6), Inches(0.5),
       '作者姓名 / Author Name', size=18, color=DARK, bold=True)
add_tb(slide, Inches(0.8), Inches(4.5), Inches(6), Inches(0.5),
       '单位 / Institution', size=14, color=GRAY)
add_tb(slide, Inches(0.8), Inches(5.0), Inches(6), Inches(0.5),
       '2026年6月 / June 2026', size=12, color=GRAY)
# Logo 占位框
logo_holder = add_rect(slide, Inches(9.5), Inches(4.5), Inches(3), Inches(1.5), fill_color=LGRAY)
add_tb(slide, Inches(10.2), Inches(5.1), Inches(1.8), Inches(0.3),
       '[LOGO]', size=11, color=GRAY, align=PP_ALIGN.CENTER)

# ============================================================
# Slide 2: 目录/大纲
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_rect(slide, Inches(0), Inches(0), W, Inches(0.12), fill_color=PRIMARY)
add_tb(slide, Inches(0.8), Inches(0.5), Inches(11), Inches(0.7),
       '目录 / Outline', size=30, color=PRIMARY, bold=True)
add_rect(slide, Inches(0.8), Inches(1.2), Inches(3), Inches(0.04), fill_color=ACCENT)
items = ['1. 研究背景与动机', '2. 数据集与预处理', '3. 模型架构',
         '4. 实验设计与结果', '5. 分析与讨论', '6. 总结与展望']
for i, item in enumerate(items):
    add_tb(slide, Inches(1.5), Inches(1.8 + i * 0.7), Inches(10), Inches(0.5),
           item, size=20, color=DARK)
add_footer(slide)

# ============================================================
# Slide 3: 正文内容页
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_rect(slide, Inches(0), Inches(0), W, Inches(0.06), fill_color=PRIMARY)
add_tb(slide, Inches(0.8), Inches(0.3), Inches(11), Inches(0.6),
       '内容页标题 / Slide Title', size=28, color=PRIMARY, bold=True)
add_rect(slide, Inches(0.8), Inches(1.0), Inches(W.inches - 0.8), Inches(0.04), fill_color=ACCENT)
add_tb(slide, Inches(0.8), Inches(1.4), Inches(11.5), Inches(5.0),
       '正文内容区域\n\n- 项目符号文本\n- 第二个要点\n- 第三个要点\n\n'
       '可插入图片、表格、公式\n\n[Content Area]', size=16, color=DARK)
add_footer(slide)

# ============================================================
# Slide 4: 图表页（两栏布局）
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
add_rect(slide, Inches(0), Inches(0), W, Inches(0.06), fill_color=PRIMARY)
add_tb(slide, Inches(0.8), Inches(0.3), Inches(11), Inches(0.6),
       '图表页 / Figure & Table', size=28, color=PRIMARY, bold=True)
add_rect(slide, Inches(0.8), Inches(1.0), Inches(W.inches - 0.8), Inches(0.04), fill_color=ACCENT)
# 左栏 - 图片占位
add_rect(slide, Inches(0.8), Inches(1.5), Inches(5.8), Inches(4.5), fill_color=LGRAY)
add_tb(slide, Inches(2.5), Inches(3.2), Inches(3), Inches(0.5),
       '[ 图表 / Figure ]', size=12, color=GRAY, align=PP_ALIGN.CENTER)
# 右栏 - 文字
add_tb(slide, Inches(7.2), Inches(1.5), Inches(5.5), Inches(4.5),
       '图表说明文字\n\n要点一\n要点二\n要点三\n\n[Description Area]', size=14, color=DARK)
add_footer(slide)

# ============================================================
# Slide 5: 章节分隔页
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
bg = slide.background
bg.fill.solid()
bg.fill.fore_color.rgb = PRIMARY
add_tb(slide, Inches(1.5), Inches(2.5), Inches(10), Inches(1.0),
       'Section Title', size=42, color=WHITE, bold=True)
add_rect(slide, Inches(1.5), Inches(3.7), Inches(2), Inches(0.05), fill_color=WHITE)
add_tb(slide, Inches(1.5), Inches(4.0), Inches(10), Inches(0.7),
       '章节副标题 / Section Subtitle', size=18, color=RGBColor(0xBB, 0xBB, 0xBB))

# ============================================================
# Slide 6: 致谢/结束页
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
bg2 = slide.background
bg2.fill.solid()
bg2.fill.fore_color.rgb = PRIMARY
add_tb(slide, Inches(2), Inches(2.5), Inches(9), Inches(1.2),
       'Thank You!', size=48, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
add_tb(slide, Inches(2), Inches(3.7), Inches(9), Inches(0.6),
       'Questions & Discussion', size=24, color=RGBColor(0xCC, 0xCC, 0xCC),
       align=PP_ALIGN.CENTER)
add_tb(slide, Inches(2), Inches(5.0), Inches(9), Inches(0.5),
       '联系方式 / Contact: email@example.com', size=14, color=GRAY,
       align=PP_ALIGN.CENTER)

# ============================================================
# 保存
# ============================================================
output_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'paper_draft', 'BIT_Beamer_Template.pptx'
)
prs.save(output_path)
print(f'Done: {output_path}')
print(f'Slides: {len(prs.slides)}')
print("""
6 张母版幻灯片:
  1. 标题页 (Title)
  2. 目录页 (Outline)
  3. 正文内容页 (Content)
  4. 图表页 (Figure/Table - 2-col)
  5. 章节分隔页 (Section Divider)
  6. 致谢页 (Thank You)

颜色: 深绿 #024A15 / 亮绿 #398207
字体: Arial (可替换为学校官方字体)
比例: 16:9
""")
