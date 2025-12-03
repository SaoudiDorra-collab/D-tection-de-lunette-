# app.py
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from datetime import datetime
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -----------------------------
# Variables globales pour les modèles
# -----------------------------
model_cnn_fast = None
model_vgg_fast = None

# -----------------------------
# Charger les modèles
# -----------------------------
def load_models():
    global model_cnn_fast, model_vgg_fast
    
    print("🔍 Chargement des modèles...")
    
    # Charger CNN Fast
    try:
        if os.path.exists("model_cnn_fast.h5"):
            model_cnn_fast = tf.keras.models.load_model("model_cnn_fast.h5")
            print("✅ Modèle CNN Fast chargé avec succès")
        else:
            print("❌ Fichier model_cnn_fast.h5 non trouvé")
    except Exception as e:
        print(f"⚠️  Erreur lors du chargement de CNN Fast: {e}")
    
    # Charger VGG Fast
    try:
        if os.path.exists("model_vgg_fast.h5"):
            model_vgg_fast = tf.keras.models.load_model("model_vgg_fast.h5")
            print("✅ Modèle VGG Fast chargé avec succès")
        else:
            print("❌ Fichier model_vgg_fast.h5 non trouvé")
    except Exception as e:
        print(f"⚠️  Erreur lors du chargement de VGG Fast: {e}")
    
    return model_cnn_fast is not None or model_vgg_fast is not None

# Charger les modèles au démarrage
load_models()

# -----------------------------
# Fonctions utilitaires
# -----------------------------
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(150, 150)):
    """Prétraitement de l'image"""
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    """Page d'accueil"""
    # Vérifier quels modèles sont disponibles
    cnn_available = model_cnn_fast is not None
    vgg_available = model_vgg_fast is not None
    
    return render_template("index.html", 
                         cnn_available=cnn_available,
                         vgg_available=vgg_available)

@app.route("/predict", methods=["POST"])
def predict():
    """Route de prédiction"""
    try:
        # Vérifier si un fichier a été envoyé
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Aucun fichier envoyé'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Aucun fichier sélectionné'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Type de fichier non autorisé. Formats acceptés: PNG, JPG, JPEG, GIF'
            }), 400
        
        # Récupérer le type de modèle
        model_type = request.form.get('model_type', 'cnn')
        
        # Choisir le modèle approprié
        if model_type == 'cnn' and model_cnn_fast is not None:
            model = model_cnn_fast
            model_name = "CNN Fast"
            target_size = (150, 150)  # Taille utilisée pour l'entraînement
        elif model_type == 'vgg' and model_vgg_fast is not None:
            model = model_vgg_fast
            model_name = "VGG16 Fast"
            target_size = (150, 150)  # Taille utilisée pour l'entraînement
        else:
            # Fallback: utiliser le premier modèle disponible
            if model_cnn_fast is not None:
                model = model_cnn_fast
                model_name = "CNN Fast"
                target_size = (150, 150)
            elif model_vgg_fast is not None:
                model = model_vgg_fast
                model_name = "VGG16 Fast"
                target_size = (150, 150)
            else:
                return jsonify({
                    'success': False,
                    'error': 'Aucun modèle disponible. Veuillez entraîner un modèle d\'abord.'
                }), 500
        
        # Générer un nom de fichier unique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Sauvegarder le fichier
        file.save(filepath)
        
        # Prétraitement de l'image
        img_array = preprocess_image(filepath, target_size)
        
        # Prédiction
        start_time = time.time()
        prediction = model.predict(img_array, verbose=0)[0][0]
        inference_time = time.time() - start_time
        
        # Interpréter le résultat
        has_glasses = bool(prediction > 0.5)
        confidence = float(prediction) if has_glasses else float(1 - prediction)
        
        # Préparer la réponse
        response = {
            'success': True,
            'has_glasses': has_glasses,
            'prediction_raw': float(prediction),
            'prediction': 'Avec Lunettes 😎' if has_glasses else 'Sans Lunettes 👓',
            'confidence': round(confidence * 100, 2),
            'model_used': model_name,
            'inference_time': round(inference_time, 3),
            'image_url': f'/static/uploads/{filename}',
            'file_size': os.path.getsize(filepath)
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Erreur lors de la prédiction: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route("/models/status")
def models_status():
    """Endpoint pour vérifier l'état des modèles"""
    return jsonify({
        'cnn_loaded': model_cnn_fast is not None,
        'vgg_loaded': model_vgg_fast is not None,
        'cnn_name': 'CNN Fast' if model_cnn_fast is not None else 'Non disponible',
        'vgg_name': 'VGG16 Fast' if model_vgg_fast is not None else 'Non disponible',
        'total_models': int(model_cnn_fast is not None) + int(model_vgg_fast is not None)
    })

@app.route("/models/reload", methods=["POST"])
def reload_models():
    """Recharge les modèles"""
    success = load_models()
    
    return jsonify({
        'success': success,
        'message': 'Modèles rechargés' if success else 'Erreur lors du rechargement',
        'cnn_loaded': model_cnn_fast is not None,
        'vgg_loaded': model_vgg_fast is not None
    })

# -----------------------------
# Route pour servir les fichiers statiques
# -----------------------------
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# -----------------------------
# Route de test simple
# -----------------------------
@app.route("/test", methods=["GET"])
def test():
    """Route de test simple"""
    return jsonify({
        'status': 'ok',
        'message': 'Serveur Flask fonctionnel',
        'models': {
            'cnn': model_cnn_fast is not None,
            'vgg': model_vgg_fast is not None
        }
    })

# -----------------------------
# Gestion des erreurs
# -----------------------------
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Page non trouvée'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Erreur interne du serveur'}), 500

# -----------------------------
# Lancement de l'application
# -----------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Application de Détection de Lunettes")
    print("=" * 60)
    print(f"📦 Modèle CNN Fast: {'✅ Chargé' if model_cnn_fast is not None else '❌ Non chargé'}")
    print(f"📦 Modèle VGG Fast: {'✅ Chargé' if model_vgg_fast is not None else '❌ Non chargé'}")
    print("=" * 60)
    
    # Vérifier le dossier uploads
    if not os.path.exists('static/uploads'):
        os.makedirs('static/uploads')
        print("📁 Dossier uploads créé: static/uploads")
    
    print("🌐 Serveur démarré sur: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)