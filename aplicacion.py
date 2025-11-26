import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from PIL import Image
import sqlite3
from datetime import datetime
import json

# Configuración de la aplicación
app = Flask(__name__)
app.secret_key = 'maiz_secret_key_2024'
app.config['SESSION_TYPE'] = 'filesystem'

# Configuración de uploads
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Crear directorios necesarios
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('database', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Base de datos para análisis
def init_database():
    conn = sqlite3.connect('database/analysis.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            filename TEXT,
            prediction TEXT,
            confidence REAL,
            disease_info TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_database()

def save_analysis(filename, prediction, confidence, disease_info):
    conn = sqlite3.connect('database/analysis.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO analysis_history (filename, prediction, confidence, disease_info)
        VALUES (?, ?, ?, ?)
    ''', (filename, prediction, confidence, json.dumps(disease_info)))
    conn.commit()
    analysis_id = cursor.lastrowid
    conn.close()
    return analysis_id

def get_analysis_history(limit=10):
    conn = sqlite3.connect('database/analysis.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, timestamp, filename, prediction, confidence, disease_info
        FROM analysis_history 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (limit,))
    
    results = []
    for row in cursor.fetchall():
        results.append({
            'id': row[0],
            'timestamp': row[1],
            'filename': row[2],
            'prediction': row[3],
            'confidence': row[4],
            'disease_info': json.loads(row[5])
        })
    conn.close()
    return results

# Información de enfermedades
DISEASE_INFO = {
    "Saludable": {
        "description": "Planta de maíz en estado óptimo sin signos de enfermedad.",
        "symptoms": "Color verde uniforme, hojas intactas, crecimiento vigoroso.",
        "treatment": "Mantener prácticas agrícolas actuales. Monitoreo preventivo.",
        "prevention": "Rotación de cultivos, uso de semillas certificadas, riego adecuado.",
        "severity": "Ninguna",
        "color": "success"
    },
    "Tizón de la Hoja": {
        "description": "Enfermedad fúngica causada por Exserohilum turcicum que afecta el follaje.",
        "symptoms": "Lesiones alargadas color gris-marrón en forma de cigarro.",
        "treatment": "Fungicidas sistémicos (estrobilurinas + triazoles).",
        "prevention": "Rotación de 2-3 años, eliminar residuos de cosecha.",
        "severity": "Alta",
        "color": "danger"
    },
    "Roya Común": {
        "description": "Enfermedad fúngica causada por Puccinia sorghi que forma pústulas.",
        "symptoms": "Pústulas pequeñas circulares color naranja-marrón.",
        "treatment": "Fungicidas protectores (clorotalonil) o sistémicos.",
        "prevention": "Evitar siembras tardías, adecuado espaciamiento.",
        "severity": "Media",
        "color": "warning"
    }
}

def analyze_image_color(image_path):
    """Análisis de color para detección de enfermedades"""
    image = cv2.imread(image_path)
    if image is None:
        return "Error", 0.0
    
    # Convertir a HSV para análisis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Rangos de color
    healthy_lower = np.array([35, 40, 40])
    healthy_upper = np.array([85, 255, 255])
    
    brown_lower = np.array([10, 50, 20])
    brown_upper = np.array([20, 255, 200])
    
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    
    # Crear máscaras
    healthy_mask = cv2.inRange(hsv, healthy_lower, healthy_upper)
    brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    
    # Calcular porcentajes
    total_pixels = image.shape[0] * image.shape[1]
    healthy_ratio = cv2.countNonZero(healthy_mask) / total_pixels
    brown_ratio = cv2.countNonZero(brown_mask) / total_pixels
    yellow_ratio = cv2.countNonZero(yellow_mask) / total_pixels
    
    # Lógica de diagnóstico
    if brown_ratio > 0.12:
        confidence = min(0.85 + brown_ratio, 0.95)
        return "Tizón de la Hoja", confidence
    elif yellow_ratio > 0.18:
        confidence = min(0.75 + yellow_ratio, 0.90)
        return "Roya Común", confidence
    elif healthy_ratio > 0.75:
        confidence = min(0.90 + healthy_ratio/10, 0.98)
        return "Saludable", confidence
    else:
        return "Saludable", 0.80

def create_annotated_image(image_path, prediction, confidence):
    """Crear imagen anotada con diagnóstico"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Redimensionar si es muy grande
    height, width = image.shape[:2]
    if width > 800:
        ratio = 800 / width
        new_width = 800
        new_height = int(height * ratio)
        image = cv2.resize(image, (new_width, new_height))
    
    # Añadir texto de diagnóstico
    text = f"{prediction} ({confidence:.1%})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Fondo para texto
    cv2.rectangle(image, (10, 10), (20 + text_size[0], 40 + text_size[1]), (40, 40, 40), -1)
    cv2.putText(image, text, (20, 40), font, font_scale, (255, 255, 255), thickness)
    
    # Guardar imagen anotada
    annotated_filename = f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.path.basename(image_path)}"
    annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
    cv2.imwrite(annotated_path, image)
    
    return annotated_filename

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No se subió ningún archivo'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Guardar archivo
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Analizar imagen
            prediction, confidence = analyze_image_color(filepath)
            
            # Crear imagen anotada
            annotated_filename = create_annotated_image(filepath, prediction, confidence)
            
            # Obtener información de la enfermedad
            disease_info = DISEASE_INFO.get(prediction, DISEASE_INFO["Saludable"])
            
            # Guardar en base de datos
            analysis_id = save_analysis(filename, prediction, confidence, disease_info)
            
            return jsonify({
                'success': True,
                'analysis_id': analysis_id,
                'prediction': prediction,
                'confidence': float(confidence),
                'disease_info': disease_info,
                'image_url': f'/static/uploads/{filename}',
                'annotated_image_url': f'/static/uploads/{annotated_filename}' if annotated_filename else f'/static/uploads/{filename}',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return jsonify({'error': f'Error procesando imagen: {str(e)}'}), 500
    
    return jsonify({'error': 'Tipo de archivo no permitido'}), 400

@app.route('/history')
def history():
    analyses = get_analysis_history(20)
    return render_template('history.html', analyses=analyses)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

if __name__ == '__main__':
    print("🌽 Iniciando Detector de Enfermedades de Maíz...")
    print("📍 URL principal: http://localhost:5000")
    print("📊 Dashboard: http://localhost:5000/dashboard")
    print("📚 Historial: http://localhost:5000/history")
    print("📷 Cámara: http://localhost:5000/camera")
    app.run(debug=True, host='0.0.0.0', port=5000)