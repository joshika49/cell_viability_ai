from flask import Flask, request, jsonify, send_file, send_from_directory
import cv2
import numpy as np
import io
import os
from fpdf import FPDF

app = Flask(__name__, static_folder='../', static_url_path='')

latest_report_data = None

@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    global latest_report_data
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "error": "No file uploaded"}), 400
            
        file = request.files['file']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"status": "error", "error": "Invalid image file"}), 400

        # --- OpenCV Processing Core ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        total_cells = len(contours)
        live_cells = 0
        dead_cells = 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            roi = img[y:y+h, x:x+w]
            b, g, r = cv2.split(roi)
            if np.mean(g) > np.mean(r):
                live_cells += 1
            else:
                dead_cells += 1

        viability_rate = round((live_cells / total_cells) * 100, 2) if total_cells > 0 else 0.0

        # --- Clinical PDF Document Structure ---
        pdf = FPDF()
        pdf.add_page()
        pdf.set_fill_color(10, 10, 10)
        pdf.rect(0, 0, 210, 40, 'F')
        
        pdf.set_font("Helvetica", "B", 24)
        pdf.set_text_color(204, 164, 59)
        pdf.text(15, 25, "VIABILITY AI - CLINICAL REPORT")
        
        pdf.set_font("Helvetica", "", 12)
        pdf.set_y(55)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(180, 10, f"Total Registered Cell Matrix: {total_cells}", ln=True)
        pdf.cell(180, 10, f"Viable Ingestion Population (Live): {live_cells}", ln=True)
        pdf.cell(180, 10, f"Necrotic Ingestion Population (Dead): {dead_cells}", ln=True)
        pdf.cell(180, 12, f"FINAL CALCULATED VIABILITY RATE: {viability_rate}%", border=1, ln=True, align='C')
        
        pdf_buffer = io.BytesIO()
        pdf_string = pdf.output(dest='S')
        pdf_buffer.write(pdf_string.encode('latin1') if isinstance(pdf_string, str) else pdf_string)
        pdf_buffer.seek(0)
        
        latest_report_data = pdf_buffer.getvalue()

        return jsonify({
            "status": "success",
            "total_count": total_cells,
            "live_count": live_cells,
            "dead_count": dead_cells,
            "viability": viability_rate
        })
        
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/api/download-report', methods=['GET'])
def download_report():
    global latest_report_data
    if latest_report_data is None:
        return "No diagnostic assay run yet.", 400
    return send_file(io.BytesIO(latest_report_data), mimetype='application/pdf', as_attachment=True, download_name='viability_assay_report.pdf')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))