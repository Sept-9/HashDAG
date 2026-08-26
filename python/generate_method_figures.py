from pathlib import Path
from math import atan2, cos, pi, sin

from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "image"
OUT.mkdir(parents=True, exist_ok=True)

FONT_DIR = Path(r"C:\Windows\Fonts")
pdfmetrics.registerFont(TTFont("Times", FONT_DIR / "times.ttf"))
pdfmetrics.registerFont(TTFont("Times-Bold", FONT_DIR / "timesbd.ttf"))
pdfmetrics.registerFont(TTFont("Times-Italic", FONT_DIR / "timesi.ttf"))
pdfmetrics.registerFont(TTFont("Times-BoldItalic", FONT_DIR / "timesbi.ttf"))

INK = HexColor("#1B2732")
MID = HexColor("#536371")
LINE = HexColor("#C8D2DA")
PANEL = HexColor("#F6F8F9")
BLUE = HexColor("#2E6E9E")
BLUE_P = HexColor("#E7F0F6")
TEAL = HexColor("#238C83")
TEAL_P = HexColor("#E5F3F1")
PURPLE = HexColor("#7051A8")
PURPLE_P = HexColor("#EEE9F6")
ORANGE = HexColor("#D96C14")
ORANGE_P = HexColor("#F9EBDF")
GREEN = HexColor("#3F8F65")
GREEN_P = HexColor("#E7F2EB")
RED = HexColor("#B84646")
RED_P = HexColor("#F7E8E8")
GREY = HexColor("#82929D")
GREY_P = HexColor("#E9EEF1")


def make_canvas(name, width, height):
    c = Canvas(str(OUT / name), pagesize=(width * inch, height * inch))
    c.setTitle(name.removesuffix(".pdf").replace("_", " ").title())
    c.setLineCap(0)
    c.setLineJoin(0)
    return c, width * inch, height * inch


def rect(c, x, y, w, h, fill=white, stroke=LINE, width=0.8):
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(width)
    c.rect(x, y, w, h, fill=1, stroke=1)


def line(c, x1, y1, x2, y2, color=INK, width=0.8, dash=None):
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.setDash(dash or [])
    c.line(x1, y1, x2, y2)
    c.setDash([])


def arrow(c, x1, y1, x2, y2, color=INK, width=0.9, head=5):
    line(c, x1, y1, x2, y2, color, width)
    a = atan2(y2 - y1, x2 - x1)
    p1 = (x2 - head * cos(a - pi / 6), y2 - head * sin(a - pi / 6))
    p2 = (x2 - head * cos(a + pi / 6), y2 - head * sin(a + pi / 6))
    line(c, x2, y2, p1[0], p1[1], color, width)
    line(c, x2, y2, p2[0], p2[1], color, width)


def text(c, value, x, y, size=9, font="Times", color=INK, align="left"):
    c.setFillColor(color)
    c.setFont(font, size)
    if align == "center":
        c.drawCentredString(x, y, value)
    elif align == "right":
        c.drawRightString(x, y, value)
    else:
        c.drawString(x, y, value)


def para(c, value, x, y, w, h, size=9, leading=None, align=TA_LEFT, color=INK, font="Times"):
    p = Paragraph(
        value,
        ParagraphStyle(
            "p",
            fontName=font,
            fontSize=size,
            leading=leading or size * 1.2,
            textColor=color,
            alignment=align,
            spaceAfter=0,
            spaceBefore=0,
        ),
    )
    _, ph = p.wrap(w, h)
    p.drawOn(c, x, y + h - ph)


def center_para(c, value, x, y, w, h, size=9, leading=None, align=TA_CENTER, color=INK, font="Times"):
    p = Paragraph(
        value,
        ParagraphStyle(
            "centered",
            fontName=font,
            fontSize=size,
            leading=leading or size * 1.2,
            textColor=color,
            alignment=align,
            spaceAfter=0,
            spaceBefore=0,
        ),
    )
    _, ph = p.wrap(w, h)
    p.drawOn(c, x, y + (h - ph) / 2)


def label(c, x, y, w, title, fill, color):
    rect(c, x, y, w, 24, fill, fill, 0)
    center_para(c, f"<b>{title}</b>", x + 5, y, w - 10, 24, 8.8, 10.5, TA_CENTER, color)


def two_line(c, title, body, cx, top_y, width, color, title_size=10, body_size=8.7):
    text(c, title, cx, top_y, title_size, "Times-Bold", color, "center")
    para(c, body, cx - width / 2, top_y - 34, width, 25, body_size, body_size + 2, TA_CENTER, MID)


def cube(c, x, y, s, fill, stroke):
    d = s * 0.25
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(0.9)
    c.rect(x, y, s, s, fill=1, stroke=1)
    c.setFillColor(white)
    p = c.beginPath()
    p.moveTo(x, y + s)
    p.lineTo(x + d, y + s + d)
    p.lineTo(x + s + d, y + s + d)
    p.lineTo(x + s, y + s)
    p.close()
    c.drawPath(p, fill=1, stroke=1)
    p = c.beginPath()
    p.moveTo(x + s, y)
    p.lineTo(x + s + d, y + d)
    p.lineTo(x + s + d, y + s + d)
    p.lineTo(x + s, y + s)
    p.close()
    c.drawPath(p, fill=1, stroke=1)


def finish(c):
    c.showPage()
    c.save()


def method_overview():
    c, w, h = make_canvas("method_overview.pdf", 12, 4.3)
    margin = 28
    label(c, margin + 16, h - 39, 142, "OFFLINE CONSTRUCTION", BLUE_P, BLUE)
    label(c, w - margin - 92, h - 39, 76, "RUNTIME", ORANGE_P, ORANGE)
    panel_x, panel_y, panel_w, panel_h = margin, 42, w - 2 * margin, h - 106
    rect(c, panel_x, panel_y, panel_w, panel_h, PANEL, LINE, 0.8)

    card_gap = 13
    card_w = (panel_w - 42 - 5 * card_gap) / 6
    card_h = 120
    card_y = panel_y + 62
    start_x = panel_x + 21
    cards = [
        ("HashDAG", "Unique geometry<br/>nodes", BLUE, BLUE_P),
        ("Bottom-up prefilter", "Colour, relief and<br/>coverage statistics", TEAL, TEAL_P),
        ("Inline descriptor", "One 32-bit word<br/>per stored node", PURPLE, PURPLE_P),
        ("Screen-space LoD", "Projected node size<br/>versus pixel footprint", ORANGE, ORANGE_P),
        ("Coverage refinement", "Dominant-axis pass bit<br/>and bounded descent", GREEN, GREEN_P),
        ("Prefiltered shading", "Representative colour<br/>and GGX response", RED, RED_P),
    ]
    for i, (title_, body_, color, fill) in enumerate(cards):
        x = start_x + i * (card_w + card_gap)
        rect(c, x, card_y, card_w, card_h, fill, color, 1.0)
        center_para(
            c,
            f"<b>{title_}</b><br/><font size='8.1'>{body_}</font>",
            x + 8,
            card_y + 62,
            card_w - 16,
            51,
            9.8,
            13,
            TA_CENTER,
            color,
        )
        if i < 5:
            arrow(c, x + card_w + 3, card_y + card_h / 2, x + card_w + card_gap - 3, card_y + card_h / 2, GREY, 0.8, 4)

        cx, cy = x + card_w / 2, card_y + 31
        if i == 0:
            nodes = [(cx, cy + 27), (cx - 18, cy + 11), (cx + 18, cy + 11), (cx - 30, cy - 6), (cx - 9, cy - 6), (cx + 9, cy - 6), (cx + 30, cy - 6)]
            for a, b in [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]:
                line(c, *nodes[a], *nodes[b], GREY, 1)
            for j, (nx, ny) in enumerate(nodes):
                c.setFillColor(BLUE if j != 4 else TEAL)
                c.circle(nx, ny, 3.5, fill=1, stroke=0)
        elif i == 1:
            vals = [10, 15, 20, 25, 30, 35]
            cols = [BLUE, TEAL, ORANGE, PURPLE, GREEN, RED]
            for j, (v, col) in enumerate(zip(vals, cols)):
                c.setFillColor(col)
                c.rect(cx - 38 + j * 13, cy - 8, 8, v, fill=1, stroke=0)
        elif i == 2:
            rect(c, cx - 39, cy - 10, 78, 31, PURPLE, PURPLE, 0)
            text(c, "32 bits", cx, cy + 1, 9, "Times-Bold", white, "center")
        elif i == 3:
            c.setStrokeColor(ORANGE)
            c.setLineWidth(1.3)
            c.circle(cx - 35, cy - 2, 5, fill=0, stroke=1)
            c.circle(cx + 28, cy + 17, 10, fill=0, stroke=1)
            arrow(c, cx - 42, cy - 7, cx + 34, cy + 20, ORANGE, 1.2, 4)
        elif i == 4:
            c.setFillColor(ORANGE)
            c.circle(cx, cy + 5, 7, fill=1, stroke=0)
            arrow(c, cx + 8, cy + 5, cx + 39, cy + 5, GREEN, 1.1, 5)
        else:
            c.setStrokeColor(RED)
            c.setLineWidth(1.2)
            c.circle(cx, cy + 4, 7, fill=0, stroke=1)
            for a in range(8):
                angle = a * pi / 4
                line(c, cx + 11 * cos(angle), cy + 4 + 11 * sin(angle), cx + 20 * cos(angle), cy + 4 + 20 * sin(angle), RED, 1.1)

    split_x = start_x + 3 * (card_w + card_gap) - card_gap / 2
    line(c, split_x, panel_y + 9, split_x, panel_y + panel_h - 9, LINE, 0.8)
    para(
        c,
        "The node descriptor is constructed once for each <b>unique DAG node</b>. "
        "At runtime it is consumed only after the projected-size test becomes eligible for termination.",
        panel_x + 32,
        panel_y + 18,
        panel_w - 64,
        24,
        8.5,
        10.5,
        TA_CENTER,
        MID,
    )
    finish(c)


def coverage_aware_lod():
    c, w, h = make_canvas("coverage_aware_lod.pdf", 12, 6.6)
    margin = 29
    gap = 17
    left_w = 428
    right_x = margin + left_w + gap
    right_w = w - margin - right_x
    rect(c, margin, 21, left_w, h - 42, PANEL, LINE, 0.8)
    rect(c, right_x, 21, right_w, h - 42, white, LINE, 0.8)
    label(c, margin + 16, h - 42, 182, "A  HIERARCHICAL RAY TRAVERSAL", BLUE_P, BLUE)
    label(c, right_x + 16, h - 42, 154, "B  TERMINATION DECISION", ORANGE_P, ORANGE)

    y_ray = 202
    c.setFillColor(HexColor("#17334B"))
    p = c.beginPath()
    p.moveTo(margin + 42, y_ray - 17)
    p.lineTo(margin + 68, y_ray)
    p.lineTo(margin + 42, y_ray + 17)
    p.close()
    c.drawPath(p, fill=1, stroke=0)
    arrow(c, margin + 68, y_ray, margin + left_w - 38, y_ray, HexColor("#17334B"), 1.3, 6)
    text(c, "ray r", margin + 84, y_ray + 40, 9, "Times-Bold")
    text(c, "camera", margin + 40, y_ray - 34, 8.6, "Times-Bold")

    cube(c, margin + 128, y_ray - 45, 76, BLUE_P, BLUE)
    cube(c, margin + 238, y_ray - 30, 48, TEAL_P, TEAL)
    cube(c, margin + 326, y_ray - 17, 28, GREEN_P, GREEN)
    text(c, "coarse", margin + 166, y_ray - 62, 8.5, "Times-Bold", BLUE, "center")
    text(c, "candidate", margin + 262, y_ray - 47, 8.5, "Times-Bold", TEAL, "center")
    text(c, "refined", margin + 340, y_ray - 34, 8.5, "Times-Bold", GREEN, "center")

    dim_x = margin + 228
    line(c, dim_x, y_ray - 31, dim_x, y_ray + 31, TEAL, 0.8)
    line(c, dim_x - 5, y_ray - 31, dim_x + 5, y_ray - 31, TEAL, 0.8)
    line(c, dim_x - 5, y_ray + 31, dim_x + 5, y_ray + 31, TEAL, 0.8)
    text(c, "S_l", dim_x - 10, y_ray - 6, 9, "Times-Bold", TEAL, "right")

    rect(c, margin + 91, 287, 126, 61, ORANGE_P, ORANGE, 0.9)
    center_para(c, "<b>Opaque enough</b><br/><font size='8.5'>terminate here</font>", margin + 99, 287, 110, 61, 9.3, 13, TA_CENTER, ORANGE)
    arrow(c, margin + 244, 239, margin + 205, 287, ORANGE, 1.1, 5)
    rect(c, margin + 249, 287, 150, 61, GREEN_P, GREEN, 0.9)
    center_para(c, "<b>Low directional coverage</b><br/><font size='8.5'>descend further</font>", margin + 256, 287, 136, 61, 9.1, 13, TA_CENTER, GREEN)
    arrow(c, margin + 273, 239, margin + 306, 287, GREEN, 1.1, 5)

    dim_y = 139
    line(c, margin + 69, dim_y, margin + 262, dim_y, GREY, 0.7, [3, 3])
    line(c, margin + 69, dim_y - 6, margin + 69, dim_y + 6, GREY, 0.7)
    line(c, margin + 262, dim_y - 6, margin + 262, dim_y + 6, GREY, 0.7)
    text(c, "distance d", margin + 165, dim_y - 16, 8.4, "Times-Italic", MID, "center")
    rect(c, margin + 34, 65, left_w - 68, 48, white, LINE, 0.8)
    para(
        c,
        "Projected size: <b><i>p</i><sub>l</sub> ≈ <i>S</i><sub>l</sub> / "
        "(<i>d</i> θ<sub>p</sub>)</b>&nbsp;&nbsp;&nbsp; "
        "Eligible for LoD when&nbsp; <b><i>p</i><sub>l</sub> &lt; τ</b>",
        margin + 44,
        77,
        left_w - 88,
        26,
        9.2,
        12,
        TA_CENTER,
    )
    para(
        c,
        "<i>S</i><sub>l</sub>: node edge length&nbsp;&nbsp; "
        "<i>d</i>: camera-to-node entry distance&nbsp;&nbsp; "
        "θ<sub>p</sub>: angular pixel footprint<br/>"
        "<i>p</i><sub>l</sub>: projected width in pixels&nbsp;&nbsp; "
        "τ: LoD termination threshold",
        margin + 38,
        28,
        left_w - 76,
        30,
        7.5,
        9,
        TA_CENTER,
        MID,
    )

    flow_x = right_x + 27
    flow_w = right_w - 54
    flow = [
        (366, 50, "Projected-size test", "<i>p</i><sub>l</sub> &lt; τ ?", BLUE, BLUE_P),
        (302, 50, "Read inline descriptor", "Only after LoD becomes eligible", PURPLE, PURPLE_P),
        (238, 50, "Choose dominant axis", "<i>a</i><super>*</super> = arg max |<i>r</i><sub>a</sub>|", TEAL, TEAL_P),
        (174, 50, "Coverage-derived passability", "<i>P</i><sub>a*</sub> = [ <i>C</i><sub>a*</sub> &lt; <i>T</i><sub>cov</sub> ]", GREEN, GREEN_P),
    ]
    for i, (y, bh, title_, body_, color, fill) in enumerate(flow):
        rect(c, flow_x, y, flow_w, bh, fill, color, 0.9)
        center_para(
            c,
            f"<b>{title_}</b><br/><font size='8.4'>{body_}</font>",
            flow_x + 8,
            y,
            flow_w - 16,
            bh,
            9.6,
            13,
            TA_CENTER,
            color,
        )
        if i < len(flow) - 1:
            arrow(c, flow_x + flow_w / 2, y, flow_x + flow_w / 2, flow[i + 1][0] + flow[i + 1][1], GREY, 0.8, 4)

    decision_y = 104
    rect(c, flow_x, decision_y, flow_w, 56, white, GREY, 0.9)
    center_para(
        c,
        "<b>Pass bit set and refinement budget remaining?</b><br/>"
        "<font size='9'><i>P</i><sub>a*</sub> = 1&nbsp;&nbsp;and&nbsp;&nbsp;"
        "<i>k</i> &lt; <i>D</i><sub>max</sub></font>",
        flow_x + 8,
        decision_y,
        flow_w - 16,
        56,
        9.2,
        14,
        TA_CENTER,
    )
    arrow(c, flow_x + flow_w / 2, 174, flow_x + flow_w / 2, decision_y + 56, GREY, 0.8, 4)

    out_gap = 12
    out_w = (flow_w - out_gap) / 2
    out_y = 34
    rect(c, flow_x, out_y, out_w, 43, GREEN_P, GREEN, 0.9)
    rect(c, flow_x + out_w + out_gap, out_y, out_w, 43, ORANGE_P, ORANGE, 0.9)
    center_para(c, "<b>YES</b><br/><font size='7.8'>descend one level</font>", flow_x + 4, out_y, out_w - 8, 43, 8.8, 12, TA_CENTER, GREEN)
    center_para(c, "<b>NO</b><br/><font size='7.8'>terminate at node</font>", flow_x + out_w + out_gap + 4, out_y, out_w - 8, 43, 8.8, 12, TA_CENTER, ORANGE)
    arrow(c, flow_x + flow_w * 0.36, decision_y, flow_x + out_w / 2, out_y + 43, GREEN, 1.0, 5)
    arrow(c, flow_x + flow_w * 0.64, decision_y, flow_x + out_w + out_gap + out_w / 2, out_y + 43, ORANGE, 1.0, 5)
    text(c, "YES", flow_x + flow_w * 0.29, 88, 7.7, "Times-Bold", GREEN, "center")
    text(c, "NO", flow_x + flow_w * 0.71, 88, 7.7, "Times-Bold", ORANGE, "center")
    finish(c)


def prefilter_construction():
    c, w, h = make_canvas("prefilter_construction.pdf", 12, 6.2)
    margin = 27
    gap = 13
    panel_w = (w - 2 * margin - 2 * gap) / 3
    panel_y = 22
    panel_h = h - 44
    xs = [margin + i * (panel_w + gap) for i in range(3)]
    for x in xs:
        rect(c, x, panel_y, panel_w, panel_h, white, LINE, 0.8)
    label(c, xs[0] + 14, h - 42, 110, "A  LEAF STATISTICS", BLUE_P, BLUE)
    label(c, xs[1] + 14, h - 42, 150, "B  BOTTOM-UP AGGREGATION", TEAL_P, TEAL)
    label(c, xs[2] + 14, h - 42, 118, "C  NODE DESCRIPTOR", PURPLE_P, PURPLE)

    para(
        c,
        "Representative slice of a <b>4 × 4 × 4 occupancy mask</b>",
        xs[0] + 25,
        368,
        panel_w - 50,
        32,
        9,
        11,
        TA_CENTER,
        MID,
    )
    gx, gy, cell = xs[0] + 39, 255, 27
    mask = [
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
    ]
    for r in range(4):
        for col in range(4):
            rect(c, gx + col * cell, gy + (3 - r) * cell, cell, cell, BLUE if mask[r][col] else white, LINE, 0.45)
    arrow(c, gx + 2 * cell, gy - 11, gx + 2 * cell, gy - 37, BLUE, 1.0, 5)
    rect(c, xs[0] + 28, 143, panel_w - 56, 70, BLUE_P, BLUE, 0.9)
    center_para(
        c,
        "Shift the mask along +/-X, +/-Y and +/-Z,<br/>"
        "then use bit operations and population counts.",
        xs[0] + 38,
        143,
        panel_w - 76,
        70,
        8.5,
        11,
        TA_CENTER,
    )
    rect(c, xs[0] + 28, 53, panel_w - 56, 64, white, LINE, 0.8)
    center_para(
        c,
        "<b><i>A</i><sub>d</sub></b>: exposed surface area<br/>"
        "<b><i>B</i><sub>d</sub></b>: occupied boundary-slab area",
        xs[0] + 38,
        53,
        panel_w - 76,
        64,
        8.8,
        12,
        TA_CENTER,
    )

    text(c, "Eight child descriptors", xs[1] + panel_w / 2, 380, 9, "Times", MID, "center")
    child_colors = [(BLUE, BLUE_P), (TEAL, TEAL_P), (ORANGE, ORANGE_P), (PURPLE, PURPLE_P)]
    child_pos = []
    for i in range(8):
        row, col = divmod(i, 4)
        cx = xs[1] + 30 + col * 58
        cy = 300 - row * 64
        color, fill = child_colors[col]
        rect(c, cx, cy, 44, 45, fill, color, 0.9)
        text(c, f"c{i}", cx + 22, cy + 17, 8.6, "Times-Bold", color, "center")
        child_pos.append((cx + 22, cy))
    agg_y = 140
    agg_x = xs[1] + 27
    agg_w = panel_w - 54
    for cx, cy in child_pos:
        line(c, cx, cy, xs[1] + panel_w / 2, agg_y + 66, GREY, 0.85)
    rect(c, agg_x, agg_y, agg_w, 66, TEAL_P, TEAL, 0.9)
    center_para(
        c,
        "Sum child areas and retain only children<br/>"
        "touching each parent boundary.",
        agg_x + 12,
        agg_y,
        agg_w - 24,
        66,
        8.7,
        11,
        TA_CENTER,
    )
    arrow(c, xs[1] + panel_w / 2, agg_y, xs[1] + panel_w / 2, 116, TEAL, 1.0, 5)
    rect(c, agg_x, 45, agg_w, 63, ORANGE_P, ORANGE, 0.9)
    center_para(
        c,
        "Subtract sibling interface overlap<br/>"
        "<b><i>I</i><sub>ij</sub> ≈ "
        "<i>B</i><sub>i</sub><i>B</i><sub>j</sub> / "
        "<i>S</i><sub>c</sub><super>2</super></b>",
        agg_x + 10,
        45,
        agg_w - 20,
        63,
        8.8,
        12,
        TA_CENTER,
    )

    para(
        c,
        "Remove the node hull already represented by the box normal:",
        xs[2] + 25,
        365,
        panel_w - 50,
        30,
        8.8,
        11,
        TA_CENTER,
        MID,
    )
    box_x = xs[2] + 29
    box_w = panel_w - 58
    rect(c, box_x, 307, box_w, 50, PURPLE_P, PURPLE, 0.9)
    center_para(
        c,
        "<b><i>r</i><sub>d</sub> = clamp(("
        "<i>A</i><sub>d</sub> - <i>B</i><sub>d</sub>) / "
        "<i>S</i><super>2</super>, 0, 1)</b>",
        box_x + 10,
        307,
        box_w - 20,
        50,
        9.2,
        12,
        TA_CENTER,
        PURPLE,
    )
    text(c, "Six-direction internal-relief histogram", xs[2] + panel_w / 2, 278, 8.6, "Times", MID, "center")
    hx, hy, hw, hh = xs[2] + 53, 224, panel_w - 106, 48
    line(c, hx, hy, hx + hw, hy, GREY, 0.7)
    vals = [20, 34, 11, 45, 24, 15]
    colors = [BLUE, BLUE, TEAL, TEAL, PURPLE, PURPLE]
    labels_ = ["-X", "+X", "-Y", "+Y", "-Z", "+Z"]
    bar_w = hw / 10
    for i, (v, col, lab) in enumerate(zip(vals, colors, labels_)):
        bx = hx + (1 + 1.45 * i) * bar_w
        c.setFillColor(col)
        c.rect(bx, hy, bar_w, v, fill=1, stroke=0)
        text(c, lab, bx + bar_w / 2, hy - 13, 7.3, "Times", MID, "center")
    arrow(c, xs[2] + panel_w / 2, 307, xs[2] + panel_w / 2, 291, PURPLE, 0.9, 4)
    arrow(c, xs[2] + panel_w / 2, 208, xs[2] + panel_w / 2, 188, GREEN, 0.9, 4)

    rect(c, box_x, 122, box_w, 62, GREEN_P, GREEN, 0.9)
    center_para(
        c,
        "<b><i>C</i><sub>a</sub> = "
        "(<i>A</i><sub>-a</sub> + <i>A</i><sub>+a</sub>) / "
        "(2<i>S</i><super>2</super>)</b>",
        box_x + 12,
        151,
        box_w - 24,
        20,
        9.0,
        11,
        TA_CENTER,
        GREEN,
    )
    center_para(
        c,
        "<b><i>P</i><sub>a</sub> = [ <i>C</i><sub>a</sub> &lt; "
        "<i>T</i><sub>cov</sub> ]</b>",
        box_x + 12,
        128,
        box_w - 24,
        20,
        9.0,
        11,
        TA_CENTER,
        GREEN,
    )
    arrow(c, xs[2] + panel_w / 2, 122, xs[2] + panel_w / 2, 107, GREY, 0.8, 4)
    rect(c, box_x, 51, box_w, 50, white, LINE, 0.8)
    center_para(
        c,
        "Memoised by <b>unique node index</b><br/>"
        "One 32-bit result is written inline.",
        box_x + 10,
        51,
        box_w - 20,
        50,
        8.6,
        11,
        TA_CENTER,
    )
    finish(c)


def storage_bar(c, x, y, widths, labels_, fills, height=36):
    cur = x
    for cw, lab, fill in zip(widths, labels_, fills):
        rect(c, cur, y, cw, height, fill, white, 0.65)
        color = white if fill not in (GREY_P,) else MID
        text(c, lab, cur + cw / 2, y + 13, 8.7, "Times-Bold", color, "center")
        cur += cw


def prefilter_layout():
    c, w, h = make_canvas("prefilter_layout.pdf", 12, 5.1)
    margin = 44
    label(c, margin, h - 42, 154, "A  HASHDAG NODE STORAGE", BLUE_P, BLUE)
    text(c, "Interior node", margin, h - 70, 9, "Times-Bold", MID)
    storage_bar(
        c,
        margin,
        h - 117,
        [91, 88, 50, 88, 125],
        ["header", "child 0", "...", "child N", "prefilter"],
        [BLUE, TEAL, GREY_P, TEAL, PURPLE],
        36,
    )
    para(
        c,
        "The trailing word is stored with the node but excluded from hashing and equality.",
        margin + 458,
        h - 105,
        w - margin * 2 - 458,
        24,
        8.7,
        10.5,
        TA_LEFT,
        MID,
    )
    text(c, "Leaf node", margin, h - 137, 9, "Times-Bold", MID)
    storage_bar(
        c,
        margin,
        h - 184,
        [145, 145, 125],
        ["occupancy [31:0]", "occupancy [63:32]", "prefilter"],
        [BLUE, TEAL, PURPLE],
        36,
    )
    para(
        c,
        "Every stored node receives exactly one additional 32-bit word.",
        margin + 432,
        h - 172,
        w - margin * 2 - 432,
        24,
        8.7,
        10.5,
        TA_LEFT,
        MID,
    )

    label(c, margin, h - 233, 170, "B  INLINE 32-BIT DESCRIPTOR", PURPLE_P, PURPLE)
    bx = margin
    by = 68
    bw = w - 2 * margin
    bh = 50
    bit_fields = [
        ("reserved", 1, GREY),
        ("Pz", 1, GREEN),
        ("Py", 1, GREEN),
        ("Px", 1, GREEN),
        ("+Z", 4, PURPLE),
        ("-Z", 4, PURPLE),
        ("+Y", 5, TEAL),
        ("-Y", 5, TEAL),
        ("+X", 5, BLUE),
        ("-X", 5, BLUE),
    ]
    widths = [bw * bits / 32 for _, bits, _ in bit_fields]
    cur = bx
    ranges = ["31", "30", "29", "28", "27..24", "23..20", "19..15", "14..10", "9..5", "4..0"]
    for (lab, bits, fill), fw, rng in zip(bit_fields, widths, ranges):
        rect(c, cur, by, fw, bh, fill, white, 0.55)
        c.saveState()
        c.setFillColor(white)
        c.setFont("Times-Bold", 7.5 if bits == 1 else 9)
        if bits == 1:
            c.translate(cur + fw / 2, by + bh / 2)
            c.rotate(90)
            c.drawCentredString(0, -2.7, lab)
        else:
            c.drawCentredString(cur + fw / 2, by + 27, lab)
            text(c, f"{bits} bits", cur + fw / 2, by + 10, 7.2, "Times", white, "center")
        c.restoreState()
        text(c, rng, cur + fw / 2, by - 14, 7.2, "Times", MID, "center")
        cur += fw
    para(
        c,
        "Relief histogram: <b>5 + 5 + 5 + 5 + 4 + 4 = 28 bits</b>&nbsp;&nbsp;&nbsp; "
        "Coverage-derived passability: <b>3 bits</b>&nbsp;&nbsp;&nbsp; Reserved: <b>1 bit</b>",
        bx,
        27,
        bw,
        23,
        8.8,
        11,
        TA_CENTER,
        MID,
    )
    finish(c)


def lod_termination():
    c, w, h = make_canvas("lod_termination.pdf", 12, 5.2)
    margin = 28
    left_w = 570
    right_x = margin + left_w + 16
    right_w = w - margin - right_x
    rect(c, margin, 22, left_w, h - 44, PANEL, LINE, 0.8)
    rect(c, right_x, 22, right_w, h - 44, white, LINE, 0.8)
    label(c, margin + 15, h - 41, 180, "A  SCREEN-SPACE FOOTPRINT", BLUE_P, BLUE)
    label(c, right_x + 15, h - 41, 160, "B  TERMINATION LEVEL", GREEN_P, GREEN)

    camera_x, camera_y = margin + 49, 180
    c.setFillColor(INK)
    p = c.beginPath()
    p.moveTo(camera_x - 13, camera_y - 12)
    p.lineTo(camera_x + 13, camera_y)
    p.lineTo(camera_x - 13, camera_y + 12)
    p.close()
    c.drawPath(p, fill=1, stroke=0)
    text(c, "camera", camera_x, camera_y - 31, 8.7, "Times-Bold", INK, "center")

    plane_x = margin + 126
    line(c, plane_x, camera_y - 54, plane_x, camera_y + 54, GREY, 1.1)
    rect(c, plane_x - 3, camera_y - 7, 6, 14, ORANGE_P, ORANGE, 0.8)
    text(c, "one pixel", plane_x, camera_y - 70, 8.2, "Times-Bold", ORANGE, "center")

    node_x, node_y, node_s = margin + 390, camera_y - 44, 88
    ray_top = camera_y + 7
    ray_bottom = camera_y - 7
    far_x = node_x + node_s + 30
    far_top = camera_y + (ray_top - camera_y) * (far_x - camera_x) / (plane_x - camera_x)
    far_bottom = camera_y + (ray_bottom - camera_y) * (far_x - camera_x) / (plane_x - camera_x)
    line(c, camera_x + 12, camera_y, far_x, camera_y, BLUE, 1.1)
    line(c, camera_x, camera_y, far_x, far_top, ORANGE, 0.85)
    line(c, camera_x, camera_y, far_x, far_bottom, ORANGE, 0.85)
    text(c, "central ray", margin + 248, camera_y + 11, 8.2, "Times-Italic", BLUE, "center")

    c.setStrokeColor(GREY)
    c.setLineWidth(0.8)
    c.setDash([4, 3])
    c.rect(node_x - 18, node_y - 18, node_s + 36, node_s + 36, fill=0, stroke=1)
    c.setDash([])
    cube(c, node_x, node_y, node_s, TEAL_P, TEAL)
    text(c, "candidate node at level l", node_x + node_s / 2, node_y + node_s + 33, 9.0, "Times-Bold", TEAL, "center")
    text(c, "parent, level l-1", node_x + node_s / 2, node_y - 33, 7.9, "Times-Italic", MID, "center")

    dim_x = node_x - 30
    line(c, dim_x, node_y, dim_x, node_y + node_s, TEAL, 0.9)
    line(c, dim_x - 5, node_y, dim_x + 5, node_y, TEAL, 0.9)
    line(c, dim_x - 5, node_y + node_s, dim_x + 5, node_y + node_s, TEAL, 0.9)
    center_para(c, "<b><i>S</i><sub>l</sub></b>", dim_x - 35, node_y + node_s / 2 - 10, 25, 20, 9.4, 11, TA_CENTER, TEAL)

    distance_y = 84
    line(c, camera_x, distance_y, node_x, distance_y, BLUE, 0.85)
    line(c, camera_x, distance_y - 5, camera_x, distance_y + 5, BLUE, 0.85)
    line(c, node_x, distance_y - 5, node_x, distance_y + 5, BLUE, 0.85)
    text(c, "distance d", (camera_x + node_x) / 2, distance_y - 15, 9.0, "Times-BoldItalic", BLUE, "center")

    footprint_x = far_x + 3
    line(c, footprint_x, far_bottom, footprint_x, far_top, ORANGE, 0.9)
    line(c, footprint_x - 5, far_bottom, footprint_x + 5, far_bottom, ORANGE, 0.9)
    line(c, footprint_x - 5, far_top, footprint_x + 5, far_top, ORANGE, 0.9)
    center_para(
        c,
        "pixel footprint<br/><b><i>d</i> &#952;<sub>p</sub></b>",
        footprint_x + 5,
        camera_y - 19,
        49,
        38,
        7.7,
        9.2,
        TA_CENTER,
        ORANGE,
    )
    c.setStrokeColor(ORANGE)
    c.setLineWidth(0.9)
    c.arc(camera_x - 4, camera_y - 21, camera_x + 52, camera_y + 21, -15, 30)
    center_para(c, "<b>&#952;<sub>p</sub></b>", camera_x + 30, camera_y + 12, 28, 20, 8.8, 10, TA_CENTER, ORANGE)

    formula_x = margin + 41
    rect(c, formula_x, 29, left_w - 82, 31, white, LINE, 0.8)
    center_para(
        c,
        "<b><i>p</i><sub>l</sub> &#8776; "
        "<i>S</i><sub>l</sub> / (<i>d</i> &#952;<sub>p</sub>)</b>"
        "&nbsp;&nbsp;&nbsp; projected node width in pixels",
        formula_x + 8,
        29,
        left_w - 98,
        31,
        9.2,
        11,
        TA_CENTER,
    )

    flow_x = right_x + 25
    flow_w = right_w - 50
    levels = [
        (246, 57, "level l-1", "p<sub>l-1</sub> &gt; &#964;", ORANGE, ORANGE_P, "descend"),
        (160, 64, "level l*", "p<sub>l</sub> &le; &#964;", GREEN, GREEN_P, "terminate"),
        (78, 52, "level l+1", "not visited", GREY, GREY_P, ""),
    ]
    for i, (y, bh, name, test, color, fill, action) in enumerate(levels):
        rect(c, flow_x, y, flow_w, bh, fill, color, 0.9)
        icon_s = 29 - i * 5
        cube(c, flow_x + 17, y + 13, icon_s, fill, color)
        para(
            c,
            f"<b>{name}</b><br/><font size='8.4'>{test}</font>",
            flow_x + 64,
            y + 10,
            flow_w - 75,
            bh - 17,
            9.3,
            12,
            TA_LEFT,
            color,
        )
        if action:
            text(c, action, flow_x + flow_w - 12, y + 11, 7.8, "Times-BoldItalic", color, "right")
        if i < 2:
            arrow(c, flow_x + flow_w / 2, y, flow_x + flow_w / 2, levels[i + 1][0] + levels[i + 1][1], GREY, 0.8, 4)
    text(c, "selected termination level", flow_x + flow_w / 2, 139, 8.2, "Times-Bold", GREEN, "center")
    rect(c, flow_x, 31, flow_w, 32, white, LINE, 0.8)
    center_para(c, "<b><i>l</i><super>*</super> = first level with "
                "<i>p</i><sub>l</sub> &le; &#964;</b>", flow_x + 7, 31, flow_w - 14, 32, 8.8, 11, TA_CENTER)
    finish(c)


def relief_values(occupied):
    directions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    bin_max = [31, 31, 31, 31, 15, 15]
    values = []
    for i, direction in enumerate(directions):
        area = 0
        boundary = 0
        for x, y, z in occupied:
            neighbour = (x + direction[0], y + direction[1], z + direction[2])
            if neighbour not in occupied:
                area += 1
            coordinate = (x, y, z)[i // 2]
            if coordinate == (0 if i % 2 == 0 else 3):
                boundary += 1
        raw = max(0.0, min(1.0, (area - boundary) / 16.0))
        values.append(round(raw * bin_max[i]) / bin_max[i])
    return values


def iso_voxel(c, cx, cy, x, y, z, palette, scale=8.0):
    sx = cx + (x - y) * scale
    sy = cy + (x + y) * scale * 0.42 + z * scale
    top, left, right = palette
    c.setStrokeColor(white)
    c.setLineWidth(0.35)
    p = c.beginPath()
    p.moveTo(sx, sy + scale)
    p.lineTo(sx + scale, sy + scale * 1.42)
    p.lineTo(sx, sy + scale * 1.84)
    p.lineTo(sx - scale, sy + scale * 1.42)
    p.close()
    c.setFillColor(top)
    c.drawPath(p, fill=1, stroke=1)
    p = c.beginPath()
    p.moveTo(sx - scale, sy + scale * 0.42)
    p.lineTo(sx, sy)
    p.lineTo(sx, sy + scale)
    p.lineTo(sx - scale, sy + scale * 1.42)
    p.close()
    c.setFillColor(left)
    c.drawPath(p, fill=1, stroke=1)
    p = c.beginPath()
    p.moveTo(sx, sy)
    p.lineTo(sx + scale, sy + scale * 0.42)
    p.lineTo(sx + scale, sy + scale * 1.42)
    p.lineTo(sx, sy + scale)
    p.close()
    c.setFillColor(right)
    c.drawPath(p, fill=1, stroke=1)


def draw_voxel_node(c, cx, cy, occupied, palette, scale=10.0):
    top, left, right = palette

    def project(point):
        x, y, z = point
        return cx + (x - y) * scale, cy - (x + y) * scale * 0.50 + z * scale

    faces = []
    definitions = [
        ((1, 0, 0), lambda x, y, z: [(x + 1, y, z), (x + 1, y + 1, z),
                                      (x + 1, y + 1, z + 1), (x + 1, y, z + 1)], right),
        ((0, 1, 0), lambda x, y, z: [(x, y + 1, z), (x + 1, y + 1, z),
                                      (x + 1, y + 1, z + 1), (x, y + 1, z + 1)], left),
        ((0, 0, 1), lambda x, y, z: [(x, y, z + 1), (x + 1, y, z + 1),
                                      (x + 1, y + 1, z + 1), (x, y + 1, z + 1)], top),
    ]
    for x, y, z in occupied:
        for direction, vertices, fill in definitions:
            neighbour = (x + direction[0], y + direction[1], z + direction[2])
            if neighbour in occupied:
                continue
            points = vertices(x, y, z)
            depth = sum(px + py + pz for px, py, pz in points) / 4.0
            faces.append((depth, [project(point) for point in points], fill))
    for _, points, fill in sorted(faces, key=lambda item: item[0]):
        p = c.beginPath()
        p.moveTo(*points[0])
        for point in points[1:]:
            p.lineTo(*point)
        p.close()
        c.setFillColor(fill)
        c.setStrokeColor(LINE)
        c.setLineWidth(0.55)
        c.drawPath(p, fill=1, stroke=1)


def draw_node_bounds(c, cx, cy, accent, show_hidden, scale=10.0):
    def project(point):
        x, y, z = point
        return cx + (x - y) * scale, cy - (x + y) * scale * 0.50 + z * scale

    points = {
        "a": project((0, 0, 4)),
        "b": project((4, 0, 4)),
        "c": project((4, 4, 4)),
        "d": project((0, 4, 4)),
        "a0": project((0, 0, 0)),
        "b0": project((4, 0, 0)),
        "c0": project((4, 4, 0)),
        "d0": project((0, 4, 0)),
    }
    if show_hidden:
        for start, end in [("c", "c0"), ("b0", "c0"), ("c0", "d0")]:
            line(c, *points[start], *points[end], GREY, 0.65, [2.5, 2.5])
    for start, end in [
        ("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"),
        ("a", "a0"), ("b", "b0"), ("d", "d0"),
        ("b0", "a0"), ("a0", "d0"),
    ]:
        line(c, *points[start], *points[end], accent, 1.0)


def relief_histogram():
    c, w, h = make_canvas("relief_histogram.pdf", 12, 5.7)
    margin = 27
    gap = 12
    panel_w = (w - 2 * margin - 3 * gap) / 4
    panel_y = 55
    panel_h = h - 78
    patterns = [
        ("(a) Solid block", {(x, y, z) for x in range(4) for y in range(4) for z in range(4)},
         (HexColor("#C7DDEB"), BLUE, HexColor("#24597F")), BLUE),
        ("(b) Flat slab", {(x, y, 1) for x in range(4) for y in range(4)},
         (HexColor("#C5E4E0"), TEAL, HexColor("#1C716A")), TEAL),
        ("(c) Staircase", {(x, y, z) for x in range(4) for y in range(4) for z in range(x + 1)},
         (HexColor("#F4D2B5"), ORANGE, HexColor("#A9510F")), ORANGE),
        ("(d) Porous node", {(x, y, z) for x in range(4) for y in range(4) for z in range(4)
                              if (x + y + z) % 2 == 0},
         (HexColor("#DDD3ED"), PURPLE, HexColor("#563D83")), PURPLE),
    ]
    labels_ = ["-X", "+X", "-Y", "+Y", "-Z", "+Z"]
    for index, (heading, occupied, palette, accent) in enumerate(patterns):
        x = margin + index * (panel_w + gap)
        rect(c, x, panel_y, panel_w, panel_h, white, LINE, 0.8)
        text(c, heading, x + panel_w / 2, panel_y + panel_h - 22, 9.4, "Times-Bold", INK, "center")
        node_y = 280
        draw_voxel_node(c, x + panel_w / 2, node_y, occupied, palette, 10.0)
        text(c, f"occupancy: {len(occupied)} / 64", x + panel_w / 2, 233, 8.0, "Times-Italic", MID, "center")

        values = relief_values(occupied)
        chart_x = x + 25
        chart_y = 93
        chart_w = panel_w - 43
        chart_h = 108
        line(c, chart_x, chart_y, chart_x, chart_y + chart_h, GREY, 0.65)
        line(c, chart_x, chart_y, chart_x + chart_w, chart_y, GREY, 0.65)
        for tick, label_ in [(0.0, "0"), (0.5, "0.5"), (1.0, "1")]:
            ty = chart_y + tick * chart_h
            line(c, chart_x - 3, ty, chart_x + chart_w, ty, LINE, 0.45, [2, 2] if tick else None)
            text(c, label_, chart_x - 6, ty - 2.5, 6.8, "Times", MID, "right")
        slot = chart_w / 6
        bar_w = slot * 0.55
        for j, (value, direction) in enumerate(zip(values, labels_)):
            bx = chart_x + j * slot + (slot - bar_w) / 2
            bh = value * chart_h
            c.setFillColor(accent)
            c.rect(bx, chart_y, bar_w, bh, fill=1, stroke=0)
            text(c, f"{value:.2f}", bx + bar_w / 2, chart_y + bh + 5, 6.5, "Times", accent, "center")
            text(c, direction, bx + bar_w / 2, chart_y - 13, 7.0, "Times-Bold", MID, "center")

    rect(c, margin + 166, 18, w - 2 * margin - 332, 25, PANEL, LINE, 0.7)
    center_para(
        c,
        "<b><i>r</i><sub>d</sub> = clamp((<i>A</i><sub>d</sub> - "
        "<i>B</i><sub>d</sub>) / 4<super>2</super>, 0, 1)</b>&nbsp;&nbsp; "
        "Bars show decoded 5/5/5/5/4/4-bit node-local values.",
        margin + 174,
        18,
        w - 2 * margin - 348,
        25,
        8.2,
        10,
        TA_CENTER,
    )
    finish(c)


if __name__ == "__main__":
    method_overview()
    coverage_aware_lod()
    prefilter_construction()
    prefilter_layout()
    lod_termination()
    relief_histogram()
    print(f"Generated method figures in {OUT}")
