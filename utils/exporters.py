import io
from typing import List, Dict, Any, Optional
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)


def export_transcript_txt(transcript: str) -> bytes:
    """Return transcript as UTF-8 encoded bytes."""
    return transcript.strip().encode("utf-8")


def export_summary_txt(summary: str) -> bytes:
    """Return summary as UTF-8 encoded bytes."""
    return summary.strip().encode("utf-8")


def export_full_report_pdf(
    title: str,
    summary: str,
    action_items: List[Dict[str, Any]],
    decisions: List[Dict[str, Any]],
    open_questions: List[Dict[str, Any]],
    source: str = "",
) -> bytes:
    """Generate a clean, professional PDF report using ReportLab."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    header_style = ParagraphStyle(
        "ReportHeader",
        parent=styles["Heading1"],
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=6,
    )
    subheader_style = ParagraphStyle(
        "ReportSubheader",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=14,
    )
    section_title_style = ParagraphStyle(
        "SectionTitle",
        parent=styles["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=14,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6,
    )
    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )
    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        fontName="Helvetica-Bold",
        textColor=colors.HexColor("#ffffff"),
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("AI Video Assistant — Report", header_style))
    doc_title = title if title else "Video Intelligence Summary"
    story.append(Paragraph(f"<b>Title:</b> {doc_title}", body_style))
    if source:
        story.append(Paragraph(f"<b>Source:</b> {source}", subheader_style))
    else:
        story.append(Spacer(1, 8))

    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=14))

    # 1. Summary Section
    story.append(Paragraph("Summary", section_title_style))
    if summary and summary.strip():
        # Split markdown lines into paragraphs
        for line in summary.strip().split("\n"):
            line_str = line.strip()
            if not line_str:
                story.append(Spacer(1, 4))
                continue
            if line_str.startswith("#"):
                clean_heading = line_str.lstrip("#").strip()
                story.append(Paragraph(f"<b>{clean_heading}</b>", section_title_style))
            elif line_str.startswith("-") or line_str.startswith("*"):
                clean_bullet = line_str.lstrip("-* ").strip()
                story.append(Paragraph(f"• {clean_bullet}", body_style))
            else:
                story.append(Paragraph(line_str, body_style))
    else:
        story.append(Paragraph("<i>No summary available.</i>", body_style))

    story.append(Spacer(1, 10))

    # 2. Action Items Table
    story.append(Paragraph("Action Items", section_title_style))
    if action_items:
        table_data = [[
            Paragraph("Task", table_header_style),
            Paragraph("Owner", table_header_style),
            Paragraph("Deadline", table_header_style),
        ]]
        for item in action_items:
            table_data.append([
                Paragraph(item.get("task", "Not specified"), table_cell_style),
                Paragraph(item.get("owner", "Not specified"), table_cell_style),
                Paragraph(item.get("deadline", "Not specified"), table_cell_style),
            ])

        t = Table(table_data, colWidths=[280, 120, 130])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.HexColor("#ffffff")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("<i>No action items were identified.</i>", body_style))

    story.append(Spacer(1, 10))

    # 3. Key Decisions Table
    story.append(Paragraph("Key Decisions", section_title_style))
    if decisions:
        table_data = [[
            Paragraph("Decision", table_header_style),
            Paragraph("Context / Reasoning", table_header_style),
        ]]
        for item in decisions:
            table_data.append([
                Paragraph(item.get("decision", ""), table_cell_style),
                Paragraph(item.get("context", ""), table_cell_style),
            ])

        t = Table(table_data, colWidths=[200, 330])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0891b2")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.HexColor("#ffffff")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("<i>No decisions were identified.</i>", body_style))

    story.append(Spacer(1, 10))

    # 4. Open Questions Table
    story.append(Paragraph("Open Questions", section_title_style))
    if open_questions:
        table_data = [[
            Paragraph("Question", table_header_style),
            Paragraph("Context", table_header_style),
        ]]
        for item in open_questions:
            table_data.append([
                Paragraph(item.get("question", ""), table_cell_style),
                Paragraph(item.get("context", ""), table_cell_style),
            ])

        t = Table(table_data, colWidths=[240, 290])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4f46e5")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f8fafc"), colors.HexColor("#ffffff")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("<i>No open questions were identified.</i>", body_style))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
