import json
from fpdf import FPDF
import io

class DatasetReportPDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 16)
        self.cell(0, 10, 'Dataset Health Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_json_report(result: dict):
    """Generates a JSON representation of the analysis report."""
    return json.dumps(result, indent=2)

def generate_pdf_report(result: dict):
    """Generates a PDF report from the analysis results."""
    pdf = DatasetReportPDF()
    pdf.add_page()
    pdf.set_font('helvetica', '', 12)
    
    # 1. Dataset Statistics
    stats = result.get("statistics", {})
    pdf.set_font('helvetica', 'B', 14)
    pdf.cell(0, 10, '1. Dataset Statistics', 0, 1, 'L')
    pdf.set_font('helvetica', '', 12)
    pdf.cell(0, 8, f"Rows: {stats.get('rows')}", 0, 1, 'L')
    pdf.cell(0, 8, f"Columns: {stats.get('columns')}", 0, 1, 'L')
    pdf.cell(0, 8, f"Numeric Features: {stats.get('numeric_feature_count')}", 0, 1, 'L')
    pdf.cell(0, 8, f"Categorical Features: {stats.get('categorical_feature_count')}", 0, 1, 'L')
    pdf.ln(5)
    
    # 2. Health Score
    score = result.get("health_score", {})
    pdf.set_font('helvetica', 'B', 14)
    pdf.cell(0, 10, f"2. Health Score: {score.get('score', 0)} / 100", 0, 1, 'L')
    pdf.set_font('helvetica', '', 12)
    breakdown = score.get("breakdown", {})
    if isinstance(breakdown, dict):
        for risk, level in breakdown.items():
            pdf.cell(0, 8, f"{risk}: {level}", 0, 1, 'L')
    pdf.ln(5)
    
    # 3. Recommendations
    recs_obj = result.get("recommendations", {})
    recs = recs_obj.get("recommendations", []) if isinstance(recs_obj, dict) else []
    pdf.set_font('helvetica', 'B', 14)
    pdf.cell(0, 10, '3. Smart Cleaning Recommendations', 0, 1, 'L')
    pdf.set_font('helvetica', '', 12)
    if not recs:
        pdf.cell(0, 8, "No recommendations found.", 0, 1, 'L')
    else:
        for r in recs:
            pdf.multi_cell(0, 8, f"- {r}", 0, 'L')
    pdf.ln(5)
    
    # 4. Baseline Model (if exists)
    model = result.get("baseline_model")

    if isinstance(model, dict) and model and "error" not in model:
        pdf.set_font('helvetica', 'B', 14)
        pdf.cell(0, 10, '4. Baseline ML Model Summary', 0, 1, 'L')
        pdf.set_font('helvetica', '', 12)
        pdf.cell(0, 8, f"Model Type: {model.get('model_type')}", 0, 1, 'L')
        if model.get("model_type") == "classification":
            pdf.cell(0, 8, f"Accuracy: {model.get('accuracy', 0):.2f}", 0, 1, 'L')
            pdf.cell(0, 8, f"F1-Score: {model.get('f1_score', 0):.2f}", 0, 1, 'L')
        else:
            pdf.cell(0, 8, f"RMSE: {model.get('rmse', 0):.2f}", 0, 1, 'L')
            pdf.cell(0, 8, f"R2 Score: {model.get('r2_score', 0):.2f}", 0, 1, 'L')
            
        pdf.cell(0, 8, "Top Features:", 0, 1, 'L')
        for f in model.get("top_features", []):
            if isinstance(f, dict):
                pdf.cell(0, 8, f"  - {f.get('feature')}: {f.get('importance', 0):.4f}", 0, 1, 'L')
            
    # Return as bytes
    return pdf.output(dest="S").encode("latin-1")
