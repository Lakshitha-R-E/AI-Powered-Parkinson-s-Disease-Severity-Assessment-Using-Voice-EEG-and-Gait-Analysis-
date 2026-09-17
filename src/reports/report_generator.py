"""
============================================================
PDF Clinical Report Generator
AI-Powered Parkinson's Disease Severity Assessment
============================================================
Generates professional clinical PDF reports using ReportLab.

Includes:
  - Patient information header
  - UPDRS score with gauge visualization
  - Severity classification with confidence
  - Voice / EEG / Gait analysis sections
  - SHAP feature importance plots
  - Treatment recommendations
  - Disclaimer
============================================================
"""

import io
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import (
        HRFlowable,
        Image as RLImage,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("[WARN] ReportLab not installed: pip install reportlab")


# ─────────────────────────────────────────────
# Color Palette
# ─────────────────────────────────────────────
BRAND_BLUE   = colors.HexColor("#1a237e")
BRAND_LIGHT  = colors.HexColor("#3f51b5")
ACCENT_GREEN = colors.HexColor("#27ae60")
ACCENT_AMBER = colors.HexColor("#f39c12")
ACCENT_RED   = colors.HexColor("#e74c3c")
LIGHT_GREY   = colors.HexColor("#f5f5f5")
MID_GREY     = colors.HexColor("#9e9e9e")
DARK_GREY    = colors.HexColor("#424242")

SEVERITY_COLORS = {
    "Mild":     ACCENT_GREEN,
    "Moderate": ACCENT_AMBER,
    "Severe":   ACCENT_RED,
}


def _severity_to_color(severity: str):
    return SEVERITY_COLORS.get(severity, MID_GREY)


# ─────────────────────────────────────────────
# Report Generator
# ─────────────────────────────────────────────

class ClinicalReportGenerator:
    """
    Generates PDF clinical reports for Parkinson's severity assessment.

    Usage:
        gen = ClinicalReportGenerator()
        gen.generate_report(
            patient_info={...},
            predictions={...},
            analysis_results={...},
            output_path="report.pdf",
        )
    """

    PAGE_W, PAGE_H = A4
    MARGIN         = 2 * cm

    def __init__(self):
        if not REPORTLAB_AVAILABLE:
            raise ImportError("Install ReportLab: pip install reportlab")
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Define custom paragraph styles."""
        self.style_title = ParagraphStyle(
            "ReportTitle",
            fontName="Helvetica-Bold",
            fontSize=22,
            textColor=BRAND_BLUE,
            alignment=TA_CENTER,
            spaceAfter=6,
        )
        self.style_subtitle = ParagraphStyle(
            "Subtitle",
            fontName="Helvetica",
            fontSize=11,
            textColor=BRAND_LIGHT,
            alignment=TA_CENTER,
            spaceAfter=12,
        )
        self.style_section = ParagraphStyle(
            "SectionHeader",
            fontName="Helvetica-Bold",
            fontSize=13,
            textColor=BRAND_BLUE,
            spaceBefore=12,
            spaceAfter=4,
        )
        self.style_body = ParagraphStyle(
            "BodyText",
            fontName="Helvetica",
            fontSize=10,
            textColor=DARK_GREY,
            leading=14,
            spaceAfter=4,
            alignment=TA_JUSTIFY,
        )
        self.style_label = ParagraphStyle(
            "Label",
            fontName="Helvetica-Bold",
            fontSize=10,
            textColor=DARK_GREY,
        )
        self.style_value = ParagraphStyle(
            "Value",
            fontName="Helvetica",
            fontSize=10,
            textColor=DARK_GREY,
        )
        self.style_disclaimer = ParagraphStyle(
            "Disclaimer",
            fontName="Helvetica-Oblique",
            fontSize=8,
            textColor=MID_GREY,
            leading=10,
            alignment=TA_JUSTIFY,
        )

    def _header_footer(self, canvas, doc):
        """Draw header and footer on each page."""
        canvas.saveState()
        W, H = A4

        # Header bar
        canvas.setFillColor(BRAND_BLUE)
        canvas.rect(0, H - 1.5 * cm, W, 1.5 * cm, fill=True, stroke=False)

        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 12)
        canvas.drawString(self.MARGIN, H - 1.1 * cm,
                          "AI-Powered Parkinson's Disease Severity Assessment")
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(W - self.MARGIN, H - 1.1 * cm, "CONFIDENTIAL — CLINICAL USE ONLY")

        # Footer
        canvas.setFillColor(MID_GREY)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(self.MARGIN, 0.8 * cm,
                          f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
                          f"Page {doc.page}")
        canvas.drawRightString(W - self.MARGIN, 0.8 * cm,
                               "This report is AI-generated and does not replace clinical diagnosis.")

        canvas.restoreState()

    def _section_header(self, text: str, icon: str = "●") -> List:
        """Return a formatted section header with horizontal rule."""
        return [
            Spacer(1, 0.3 * cm),
            Paragraph(f"{icon} {text}", self.style_section),
            HRFlowable(width="100%", thickness=0.5, color=BRAND_LIGHT),
            Spacer(1, 0.2 * cm),
        ]

    def _patient_info_table(self, patient_info: Dict) -> Table:
        """Build a styled patient information table."""
        rows = [
            ["Patient Name",   patient_info.get("name",       "N/A"),
             "Patient ID",     patient_info.get("patient_id", "N/A")],
            ["Age",            str(patient_info.get("age", "N/A")),
             "Gender",         patient_info.get("gender", "N/A")],
            ["Assessment Date", patient_info.get("date", datetime.now().strftime("%Y-%m-%d")),
             "Clinician",      patient_info.get("clinician", "N/A")],
            ["Diagnosis",      patient_info.get("diagnosis", "Parkinson's Disease"),
             "Report ID",      patient_info.get("report_id", "AUTO-001")],
        ]

        table = Table(rows, colWidths=[3.5*cm, 5.5*cm, 3.5*cm, 5.5*cm])
        table.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (0, -1), LIGHT_GREY),
            ("BACKGROUND",  (2, 0), (2, -1), LIGHT_GREY),
            ("FONTNAME",    (0, 0), (-1, -1), "Helvetica"),
            ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME",    (2, 0), (2, -1), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("TEXTCOLOR",   (0, 0), (-1, -1), DARK_GREY),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, LIGHT_GREY]),
            ("BOX",         (0, 0), (-1, -1), 0.5, BRAND_LIGHT),
            ("INNERGRID",   (0, 0), (-1, -1), 0.25, MID_GREY),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",  (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))
        return table

    def _prediction_summary_table(self, predictions: Dict) -> Table:
        """Build prediction summary table."""
        severity       = predictions.get("severity_label", "Unknown")
        updrs_score    = predictions.get("updrs_score", 0.0)
        confidence     = predictions.get("confidence", 0.0)
        class_probs    = predictions.get("class_probabilities", [0.33, 0.33, 0.34])

        sev_color = _severity_to_color(severity)

        rows = [
            ["Metric", "Value", "Interpretation"],
            ["UPDRS Total Score",
             f"{float(updrs_score):.1f} / 108",
             self._updrs_interpretation(float(updrs_score))],
            ["Severity Classification",
             severity,
             f"{float(confidence)*100:.1f}% confidence"],
            ["Mild Probability",
             f"{float(class_probs[0])*100:.1f}%",
             "Low disease burden"],
            ["Moderate Probability",
             f"{float(class_probs[1])*100:.1f}%",
             "Moderate disease burden"],
            ["Severe Probability",
             f"{float(class_probs[2])*100:.1f}%",
             "High disease burden"],
        ]

        table = Table(rows, colWidths=[5.5*cm, 5.5*cm, 7*cm])
        table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  BRAND_BLUE),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",      (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
            ("BOX",           (0, 0), (-1, -1), 0.5, BRAND_LIGHT),
            ("INNERGRID",     (0, 0), (-1, -1), 0.25, MID_GREY),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            # Highlight severity row
            ("BACKGROUND",    (1, 2), (1, 2),   sev_color),
            ("TEXTCOLOR",     (1, 2), (1, 2),   colors.white),
            ("FONTNAME",      (1, 2), (1, 2),   "Helvetica-Bold"),
        ]))
        return table

    @staticmethod
    def _updrs_interpretation(score: float) -> str:
        if score < 20:
            return "Minimal impairment"
        elif score < 40:
            return "Mild impairment"
        elif score < 60:
            return "Moderate impairment"
        elif score < 80:
            return "Significant impairment"
        else:
            return "Severe impairment"

    def _embed_image(self, img_path: str, width: float = 8*cm) -> Optional[RLImage]:
        """Embed an image into the PDF."""
        if img_path and Path(img_path).exists():
            aspect = 1.0
            try:
                with Image.open(img_path) as img:
                    aspect = img.height / img.width
            except Exception:
                pass
            return RLImage(img_path, width=width, height=width * aspect)
        return None

    def _feature_table(self, features: Dict[str, float], title: str) -> List:
        """Build a formatted feature values table."""
        rows = [["Feature", "Value"]]
        for name, val in list(features.items())[:12]:
            clean_name = name.replace("_", " ").title()
            rows.append([clean_name, f"{float(val):.4f}" if isinstance(val, float) else str(val)])

        table = Table(rows, colWidths=[9*cm, 9*cm])
        table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  BRAND_LIGHT),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
            ("BOX",           (0, 0), (-1, -1), 0.5, BRAND_LIGHT),
            ("INNERGRID",     (0, 0), (-1, -1), 0.25, MID_GREY),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ]))
        return [table]

    def _recommendations(self, severity: str) -> List[Paragraph]:
        """Generate treatment recommendations based on severity."""
        recs = {
            "Mild": [
                "• Initiate or continue levodopa therapy at low doses; monitor response.",
                "• Enroll in structured physiotherapy (2–3 sessions/week) targeting balance and gait.",
                "• Speech therapy evaluation for voice tremor and dysarthria screening.",
                "• Regular aerobic exercise (walking, cycling) shown to slow motor decline.",
                "• Follow-up assessment in 3–6 months.",
                "• Patient education: dietary guidance, fall prevention, sleep hygiene.",
            ],
            "Moderate": [
                "• Optimize dopaminergic therapy; consider dopamine agonists or MAO-B inhibitors.",
                "• Intensive physiotherapy (3–5 sessions/week) including balance and gait training.",
                "• Occupational therapy for ADL (activities of daily living) support.",
                "• Speech-language therapy for dysarthria management.",
                "• Consider deep brain stimulation (DBS) candidacy evaluation.",
                "• Multidisciplinary clinic referral (neurology, PT, OT, SLP).",
                "• Follow-up in 1–3 months; adjust medications as needed.",
            ],
            "Severe": [
                "• Urgent neurologist consultation for advanced medication management.",
                "• Evaluate for DBS surgery or continuous infusion therapies (apomorphine/levodopa-carbidopa).",
                "• Full-time physiotherapy and occupational therapy support.",
                "• Nutritional assessment — dysphagia screening essential.",
                "• Palliative care consultation for quality of life management.",
                "• Carer support services and home safety assessment.",
                "• Monthly follow-up; consider hospitalization if mobility significantly impaired.",
            ],
        }

        texts = recs.get(severity, recs["Moderate"])
        elements = []
        for rec in texts:
            elements.append(Paragraph(rec, self.style_body))
        return elements

    def generate_report(
        self,
        patient_info:    Dict,
        predictions:     Dict,
        analysis_results: Dict,
        output_path:     str = "parkinson_report.pdf",
        image_paths:     Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Generate complete clinical PDF report.

        Args:
            patient_info:     Patient demographic information
            predictions:      Model prediction results
            analysis_results: Voice/EEG/Gait analysis features
            output_path:      Output PDF file path
            image_paths:      Dict of plot image paths {
                                'shap_voice', 'shap_eeg', 'shap_gait',
                                'fusion_attention', 'waveform', 'eeg_psd'
                              }

        Returns:
            Path to generated PDF file
        """
        output_path = str(output_path)
        image_paths = image_paths or {}

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            topMargin=2*cm,
            bottomMargin=1.5*cm,
            leftMargin=self.MARGIN,
            rightMargin=self.MARGIN,
        )

        story = []

        # ── TITLE ─────────────────────────────────────────
        story.append(Spacer(1, 0.5*cm))
        story.append(Paragraph("Parkinson's Disease Severity Assessment Report",
                                self.style_title))
        story.append(Paragraph(
            "AI-Powered Multimodal Clinical Analysis System",
            self.style_subtitle
        ))
        story.append(HRFlowable(width="100%", thickness=1.5, color=BRAND_BLUE))
        story.append(Spacer(1, 0.4*cm))

        # ── PATIENT INFORMATION ───────────────────────────
        story.extend(self._section_header("Patient Information", "👤"))
        story.append(self._patient_info_table(patient_info))
        story.append(Spacer(1, 0.4*cm))

        # ── PREDICTION SUMMARY ────────────────────────────
        story.extend(self._section_header("Assessment Results", "📊"))
        story.append(self._prediction_summary_table(predictions))
        story.append(Spacer(1, 0.3*cm))

        severity = predictions.get("severity_label", "Moderate")
        sev_color = _severity_to_color(severity)

        story.append(Paragraph(
            f"<b>Overall Classification:</b> "
            f"<font color='{sev_color.hexval() if hasattr(sev_color, 'hexval') else '#000000'}'>"
            f"{severity} Parkinson's Disease</font>",
            self.style_body,
        ))

        # ── VOICE ANALYSIS ────────────────────────────────
        story.append(PageBreak())
        story.extend(self._section_header("Voice Signal Analysis", "🎤"))

        voice_feats = analysis_results.get("voice", {})
        if voice_feats:
            story.extend(self._feature_table(voice_feats, "Voice Features"))

        if "waveform" in image_paths:
            img = self._embed_image(image_paths["waveform"], width=16*cm)
            if img:
                story.append(Spacer(1, 0.3*cm))
                story.append(img)

        if "shap_voice" in image_paths:
            story.extend(self._section_header("Voice Feature Importance (SHAP)", "🔍"))
            img = self._embed_image(image_paths["shap_voice"], width=16*cm)
            if img:
                story.append(img)

        # ── EEG ANALYSIS ──────────────────────────────────
        story.append(PageBreak())
        story.extend(self._section_header("EEG Signal Analysis", "🧠"))

        eeg_feats = analysis_results.get("eeg", {})
        if eeg_feats:
            story.extend(self._feature_table(eeg_feats, "EEG Features"))

        if "eeg_psd" in image_paths:
            story.append(Spacer(1, 0.3*cm))
            img = self._embed_image(image_paths["eeg_psd"], width=16*cm)
            if img:
                story.append(img)

        if "shap_eeg" in image_paths:
            story.extend(self._section_header("EEG Feature Importance (SHAP)", "🔍"))
            img = self._embed_image(image_paths["shap_eeg"], width=16*cm)
            if img:
                story.append(img)

        # ── GAIT ANALYSIS ─────────────────────────────────
        story.append(PageBreak())
        story.extend(self._section_header("Gait Signal Analysis", "🚶"))

        gait_feats = analysis_results.get("gait", {})
        if gait_feats:
            story.extend(self._feature_table(gait_feats, "Gait Features"))

        if "shap_gait" in image_paths:
            story.extend(self._section_header("Gait Feature Importance (SHAP)", "🔍"))
            img = self._embed_image(image_paths["shap_gait"], width=16*cm)
            if img:
                story.append(img)

        # ── FUSION ATTENTION ──────────────────────────────
        if "fusion_attention" in image_paths:
            story.extend(self._section_header("Cross-Modal Attention Analysis", "🔗"))
            img = self._embed_image(image_paths["fusion_attention"], width=10*cm)
            if img:
                story.append(img)
            story.append(Paragraph(
                "The attention heatmap shows the relative importance of each modality "
                "(Voice, EEG, Gait) in the fusion decision-making process. "
                "Darker cells indicate stronger cross-modal attention.",
                self.style_body,
            ))

        # ── TREATMENT RECOMMENDATIONS ─────────────────────
        story.append(PageBreak())
        story.extend(self._section_header("Clinical Recommendations", "💊"))
        story.extend(self._recommendations(severity))

        # ── DISCLAIMER ────────────────────────────────────
        story.append(Spacer(1, 1*cm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GREY))
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph(
            "DISCLAIMER: This report is generated by an AI system for research purposes. "
            "It is not a substitute for professional medical diagnosis or clinical judgment. "
            "All predictions should be interpreted by a qualified neurologist in conjunction "
            "with comprehensive clinical evaluation. The AI model may have limitations "
            "depending on data quality and patient population.",
            self.style_disclaimer,
        ))

        # ── BUILD PDF ─────────────────────────────────────
        doc.build(story, onFirstPage=self._header_footer, onLaterPages=self._header_footer)
        print(f"✅ Report generated: {output_path}")
        return output_path

    def generate_report_bytes(self, **kwargs) -> bytes:
        """Generate report and return as bytes (for Streamlit download)."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            tmp_path = f.name
        self.generate_report(output_path=tmp_path, **kwargs)
        with open(tmp_path, "rb") as f:
            data = f.read()
        os.unlink(tmp_path)
        return data
