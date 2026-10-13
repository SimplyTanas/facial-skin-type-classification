#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: generate_chapter3_docx.py
Mục đích: Chuyển đổi nội dung Chương 3 (chuong3_khao_sat_du_lieu.md) thành tệp Word (.docx)
chuẩn học thuật, định dạng chuyên nghiệp theo quy chuẩn báo cáo tốt nghiệp / đồ án môn học.

Quy chuẩn định dạng:
- Phông chữ: Times New Roman
- Cỡ chữ thân bài: 13pt, dãn dòng 1.5 lines, dãn đoạn sau 6pt
- Tiêu đề Chương (Heading 1): 16pt đậm, căn giữa, dãn đoạn trước 18pt, sau 12pt
- Tiêu đề mục cấp 2 (Heading 2): 14pt đậm, dãn đoạn trước 14pt, sau 6pt
- Tiêu đề mục cấp 3 (Heading 3): 13pt đậm, dãn đoạn trước 10pt, sau 4pt
- Tiêu đề mục cấp 4 (Heading 4): 13pt đậm nghiêng, dãn đoạn trước 8pt, sau 3pt
- Căn lề trang: Trái 3.0 cm, Phải 2.0 cm, Trên 2.0 cm, Dưới 2.0 cm
- Bảng biểu: Có đường viền tinh tế, dòng tiêu đề nền xanh đậm (#1F4E79), chữ trắng đậm,
  xen kẽ nền hàng (#FFFFFF và #F8F9FA), căn lề ô chuẩn, tiêu đề bảng in nghiêng đậm.
- Khối mã / Sơ đồ ASCII: Đặt trong khung viền (Callout box) với phông Consolas 10pt.
"""

import os
import re
import docx
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls


def set_cell_background(cell, fill_hex: str):
    """Thiết lập màu nền cho ô trong bảng."""
    tcPr = cell._tc.get_or_add_tcPr()
    # Xóa shading cũ nếu có
    for child in list(tcPr):
        if child.tag.endswith('shd'):
            tcPr.remove(child)
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=140, bottom=140, left=180, right=180):
    """Thiết lập khoảng đệm (padding) trong ô bảng (đơn vị: dxa)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)


def set_table_borders(table, color="B0C4DE", sz="4", val="single"):
    """Thiết lập đường viền bảng thanh nhã."""
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:left w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:right w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)


def set_repeat_header(row):
    """Cấu hình lặp lại dòng tiêu đề khi bảng nhảy sang trang mới."""
    trPr = row._tr.get_or_add_trPr()
    trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
    trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))


def set_cant_split(row):
    """Ngăn hàng của bảng bị cắt đôi qua 2 trang."""
    trPr = row._tr.get_or_add_trPr()
    trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))


def format_run_font(run, font_name="Times New Roman", size_pt=13, bold=False, italic=False, color_rgb=None):
    """Định dạng phông chữ, kích thước, kiểu dáng cho một run."""
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    run.bold = bold
    run.italic = italic
    if color_rgb:
        run.font.color.rgb = color_rgb


def add_formatted_text(p, text: str, default_size=13, default_color=None, is_heading=False):
    """
    Phân tích chuỗi markdown chứa **bold**, *italic*, ***bold_italic***, `code`, $math$
    và thêm các run định dạng tương ứng vào đoạn văn bản p.
    """
    pattern = re.compile(r'(\*\*\*[^*]+\*\*\*|\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\$[^$]+\$)')
    tokens = pattern.split(text)

    for token in tokens:
        if not token:
            continue
        run = p.add_run()
        run.font.name = "Times New Roman"
        run.font.size = Pt(default_size)
        if default_color:
            run.font.color.rgb = default_color

        if token.startswith('***') and token.endswith('***') and len(token) > 6:
            run.text = token[3:-3]
            run.bold = True
            run.italic = True
        elif token.startswith('**') and token.endswith('**') and len(token) > 4:
            run.text = token[2:-2]
            run.bold = True
        elif token.startswith('*') and token.endswith('*') and len(token) > 2:
            run.text = token[1:-1]
            run.italic = True
        elif token.startswith('`') and token.endswith('`') and len(token) > 2:
            run.text = token[1:-1]
            run.font.name = "Consolas"
            run.font.size = Pt(default_size - 1.5 if default_size > 10 else default_size)
            run.font.color.rgb = RGBColor(180, 40, 40)
        elif token.startswith('$') and token.endswith('$') and len(token) > 2:
            run.text = token[1:-1]
            run.font.name = "Cambria Math"
            run.italic = True
        else:
            run.text = token


def create_code_callout_box(doc, code_lines):
    """
    Tạo khối hiển thị mã nguồn / sơ đồ cây ASCII trong khung viền callout sang trọng.
    """
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    cell = table.cell(0, 0)
    cell.width = Inches(6.3)
    set_cell_background(cell, "F4F6F9")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)

    # Viền bên trái đậm màu xanh primary, các viền khác mờ
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="4" w:space="0" w:color="D0D7DE"/>'
        f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="D0D7DE"/>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="1F4E79"/>'
        f'<w:right w:val="single" w:sz="4" w:space="0" w:color="D0D7DE"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)

    p = cell.paragraphs[0]
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)

    full_code = "\n".join(code_lines)
    run = p.add_run(full_code)
    run.font.name = "Consolas"
    run.font.size = Pt(9.5)
    run.font.color.rgb = RGBColor(36, 41, 47)

    # Thêm đoạn trống nhỏ sau bảng
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_before = Pt(0)
    spacer.paragraph_format.space_after = Pt(6)


def create_markdown_table(doc, table_lines):
    """
    Chuyển đổi khối bảng Markdown thành bảng Word chuyên nghiệp có màu sắc và định dạng chuẩn.
    """
    # Lọc bỏ dòng trống
    raw_rows = [line.strip() for line in table_lines if line.strip()]
    if len(raw_rows) < 2:
        return

    # Tách dữ liệu tiêu đề và các dòng
    def parse_cells(row_str):
        # Bỏ dấu | ở đầu và cuối
        content = row_str.strip()
        if content.startswith('|'):
            content = content[1:]
        if content.endswith('|'):
            content = content[:-1]
        return [c.strip() for c in content.split('|')]

    headers = parse_cells(raw_rows[0])
    num_cols = len(headers)

    # Xác định căn lề của các cột từ dòng phân cách (| :--- | :---: | ---: |)
    alignments = []
    data_start_idx = 1
    if len(raw_rows) > 1 and re.match(r'^[\|\s\:\-]+$', raw_rows[1]):
        sep_cells = parse_cells(raw_rows[1])
        data_start_idx = 2
        for sep in sep_cells:
            if sep.startswith(':') and sep.endswith(':'):
                alignments.append(WD_ALIGN_PARAGRAPH.CENTER)
            elif sep.endswith(':'):
                alignments.append(WD_ALIGN_PARAGRAPH.RIGHT)
            else:
                alignments.append(WD_ALIGN_PARAGRAPH.LEFT)
    else:
        alignments = [WD_ALIGN_PARAGRAPH.LEFT] * num_cols

    # Bổ sung nếu thiếu
    while len(alignments) < num_cols:
        alignments.append(WD_ALIGN_PARAGRAPH.LEFT)

    data_rows = []
    for r in raw_rows[data_start_idx:]:
        cells = parse_cells(r)
        if len(cells) < num_cols:
            cells += [''] * (num_cols - len(cells))
        elif len(cells) > num_cols:
            cells = cells[:num_cols]
        data_rows.append(cells)

    # Tạo bảng trong tài liệu
    table = doc.add_table(rows=len(data_rows) + 1, cols=num_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    set_table_borders(table, color="CBD5E1", sz="6", val="single")

    # 1. Định dạng dòng Tiêu đề (Header Row)
    hdr_row = table.rows[0]
    set_repeat_header(hdr_row)
    for col_idx, text in enumerate(headers):
        cell = hdr_row.cells[col_idx]
        set_cell_background(cell, "1F4E79")  # Màu xanh đậm Navy
        set_cell_margins(cell, top=140, bottom=140, left=150, right=150)
        p = cell.paragraphs[0]
        p.alignment = alignments[col_idx]
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.15
        add_formatted_text(p, text, default_size=11.5, default_color=RGBColor(255, 255, 255), is_heading=True)

    # 2. Định dạng các dòng dữ liệu (Data Rows)
    for row_idx, row_data in enumerate(data_rows):
        tbl_row = table.rows[row_idx + 1]
        set_cant_split(tbl_row)
        bg_color = "F8FAFC" if (row_idx % 2 == 1) else "FFFFFF"

        for col_idx, text in enumerate(row_data):
            cell = tbl_row.cells[col_idx]
            set_cell_background(cell, bg_color)
            set_cell_margins(cell, top=120, bottom=120, left=150, right=150)
            p = cell.paragraphs[0]
            p.alignment = alignments[col_idx]
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.2
            add_formatted_text(p, text, default_size=11, default_color=RGBColor(30, 41, 59))

    # Thêm khoảng đệm nhỏ sau bảng
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_before = Pt(0)
    spacer.paragraph_format.space_after = Pt(6)


def convert_markdown_to_docx(md_path: str, docx_path: str):
    """
    Hàm chính: Đọc tệp Markdown và chuyển đổi thành tài liệu Word định dạng cao cấp.
    """
    print(f"[*] Đang đọc tệp nguồn Markdown: {md_path}")
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"Không tìm thấy tệp markdown tại {md_path}")

    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    doc = docx.Document()

    # Cấu hình kích thước trang A4 & Căn lề chuẩn luận văn
    for section in doc.sections:
        section.page_width = Cm(21.0)
        section.page_height = Cm(29.7)
        section.left_margin = Cm(3.0)    # Lề trái 3.0 cm đóng gáy
        section.right_margin = Cm(2.0)   # Lề phải 2.0 cm
        section.top_margin = Cm(2.0)     # Lề trên 2.0 cm
        section.bottom_margin = Cm(2.0)  # Lề dưới 2.0 cm

    # Màu sắc nhận diện hệ thống
    NAVY_BLUE = RGBColor(31, 78, 121)    # #1F4E79 (Heading 1 & 2)
    SLATE_DARK = RGBColor(32, 55, 100)   # #203764 (Heading 3)
    CHARCOAL = RGBColor(33, 37, 41)      # Thân bài

    i = 0
    total_lines = len(lines)

    while i < total_lines:
        line = lines[i].rstrip('\r\n')
        stripped = line.strip()

        # 1. Bỏ qua dòng phân cách ngang markdown '---'
        if stripped == '---':
            i += 1
            continue

        # 2. Xử lý khối mã nguồn / ASCII diagram ```
        if stripped.startswith('```'):
            code_block_lines = []
            i += 1
            while i < total_lines and not lines[i].strip().startswith('```'):
                code_block_lines.append(lines[i].rstrip('\r\n'))
                i += 1
            if i < total_lines:
                i += 1  # Bỏ qua dòng kết thúc ```
            create_code_callout_box(doc, code_block_lines)
            continue

        # 3. Xử lý Bảng biểu Markdown (| Col1 | Col2 | ...)
        if stripped.startswith('|') and stripped.endswith('|'):
            table_lines = []
            while i < total_lines and lines[i].strip().startswith('|') and lines[i].strip().endswith('|'):
                table_lines.append(lines[i].strip())
                i += 1
            create_markdown_table(doc, table_lines)
            continue

        # 4. Xử lý Tiêu đề Bảng / Hình ảnh (Bắt đầu bằng *Bảng hoặc Bảng)
        if re.match(r'^\*?(Bảng|Hình)\s+\d+[\.\d]*\..*\*?$', stripped):
            clean_title = stripped.strip('*_')
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(clean_title)
            format_run_font(run, font_name="Times New Roman", size_pt=12, bold=True, italic=True, color_rgb=NAVY_BLUE)
            i += 1
            continue

        # 5. Xử lý Công thức toán hiển thị độc lập ($$ ... $$)
        if stripped.startswith('$$') and stripped.endswith('$$') and len(stripped) > 4:
            formula = stripped[2:-2].strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(8)
            p.paragraph_format.line_spacing = 1.3
            run = p.add_run(formula)
            format_run_font(run, font_name="Cambria Math", size_pt=12.5, italic=True)
            i += 1
            continue

        # 6. Xử lý Tiêu đề Heading 1 (# CHƯƠNG ...)
        if stripped.startswith('# '):
            title_text = stripped[2:].strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(20)
            p.paragraph_format.space_after = Pt(14)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(title_text)
            format_run_font(run, font_name="Times New Roman", size_pt=16, bold=True, color_rgb=NAVY_BLUE)
            i += 1
            continue

        # 7. Xử lý Tiêu đề Heading 2 (## 3.1. ...)
        if stripped.startswith('## '):
            title_text = stripped[3:].strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(16)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.keep_with_next = True
            add_formatted_text(p, title_text, default_size=14, default_color=NAVY_BLUE, is_heading=True)
            for r in p.runs:
                r.bold = True
            i += 1
            continue

        # 8. Xử lý Tiêu đề Heading 3 (### 3.1.1. ...)
        if stripped.startswith('### '):
            title_text = stripped[4:].strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            add_formatted_text(p, title_text, default_size=13, default_color=SLATE_DARK, is_heading=True)
            for r in p.runs:
                r.bold = True
            i += 1
            continue

        # 9. Xử lý Tiêu đề Heading 4 (#### a) ...)
        if stripped.startswith('#### '):
            title_text = stripped[5:].strip()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.keep_with_next = True
            add_formatted_text(p, title_text, default_size=13, default_color=CHARCOAL, is_heading=True)
            for r in p.runs:
                r.bold = True
                r.italic = True
            i += 1
            continue

        # 10. Xử lý Danh sách không thứ tự (Bullet points: - hoặc *)
        bullet_match = re.match(r'^(\s*)[-\*]\s+(.+)$', line)
        if bullet_match:
            indent_spaces = len(bullet_match.group(1))
            bullet_text = bullet_match.group(2).strip()

            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.line_spacing = 1.3
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(3)

            base_indent = 0.6 + (indent_spaces // 2) * 0.4
            p.paragraph_format.left_indent = Cm(base_indent)
            p.paragraph_format.first_line_indent = Cm(-0.4)

            # Biểu tượng dấu đầu dòng
            sym_run = p.add_run("•  ")
            format_run_font(sym_run, font_name="Times New Roman", size_pt=12, bold=True, color_rgb=NAVY_BLUE)

            add_formatted_text(p, bullet_text, default_size=13, default_color=CHARCOAL)
            i += 1
            continue

        # 11. Xử lý Danh sách có thứ tự (Numbered list: 1. , 2. )
        num_match = re.match(r'^(\s*)(\d+)[\.\)]\s+(.+)$', line)
        if num_match:
            indent_spaces = len(num_match.group(1))
            num_str = num_match.group(2)
            item_text = num_match.group(3).strip()

            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.line_spacing = 1.35
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(3)

            base_indent = 0.8 + (indent_spaces // 2) * 0.4
            p.paragraph_format.left_indent = Cm(base_indent)
            p.paragraph_format.first_line_indent = Cm(-0.5)

            num_run = p.add_run(f"{num_str}. ")
            format_run_font(num_run, font_name="Times New Roman", size_pt=13, bold=True, color_rgb=NAVY_BLUE)

            add_formatted_text(p, item_text, default_size=13, default_color=CHARCOAL)
            i += 1
            continue

        # 12. Dòng trống
        if not stripped:
            i += 1
            continue

        # 13. Đoạn văn bản thường (Body Paragraph)
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.line_spacing = 1.5           # Dãn dòng chuẩn 1.5 lines
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(6)          # Dãn đoạn sau 6pt
        p.paragraph_format.first_line_indent = Cm(1.0)  # Thụt đầu dòng 1.0 cm

        add_formatted_text(p, stripped, default_size=13, default_color=CHARCOAL)
        i += 1

    # Lưu tài liệu Word
    print(f"[*] Đang lưu tài liệu Word tại: {docx_path}")
    doc.save(docx_path)
    print(f"[✓] Đã tạo thành công tài liệu: {docx_path}")


if __name__ == "__main__":
    docs_dir = os.path.dirname(os.path.abspath(__file__))
    md_file = os.path.join(docs_dir, "chuong3_khao_sat_du_lieu.md")
    docx_file = os.path.join(docs_dir, "chuong3_khao_sat_du_lieu.docx")

    convert_markdown_to_docx(md_file, docx_file)
