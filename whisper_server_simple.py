#!/usr/bin/env python3
"""
Simple Whisper Server for Arabic Speech Recognition
Using OpenAI Whisper for reliable deployment
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import tempfile
import os
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model - load once at startup
model = None

def load_model():
    """Load OpenAI Whisper model"""
    global model
    try:
        logger.info("Loading OpenAI Whisper model (base)...")
        # Use base model for good balance of speed and accuracy
        model = whisper.load_model("base")
        logger.info("✅ Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return False

@app.route('/inference', methods=['POST'])
def transcribe():
    """
    Transcribe audio file to text
    Expects: multipart/form-data with 'file' field containing audio
    Returns: JSON with 'text' field containing transcription
    """
    try:
        # Check if file is provided
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        audio_file = request.files['file']
        
        if audio_file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Get language from query params (default: Arabic)
        language = request.args.get('language', 'ar')
        
        logger.info(f"🎤 Transcribing audio (language: {language})...")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Transcribe with OpenAI Whisper
            result = model.transcribe(
                tmp_path, 
                language=language,
                fp16=False  # Use fp32 for better compatibility
            )
            
            # Extract text
            text = result["text"].strip()
            
            logger.info(f"✅ Transcription: \"{text}\"")
            logger.info(f"   Language: {result.get('language', 'unknown')}")
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            return jsonify({
                "text": text,
                "language": result.get('language', language),
                "language_probability": 1.0  # OpenAI Whisper doesn't provide this
            })
            
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise e
            
    except Exception as e:
        logger.error(f"❌ Transcription error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "model_name": "openai-whisper-base"
    })

if __name__ == '__main__':
    print("="*50)
    print("🎤 Simple Whisper Server for Arabic Learning")
    print("="*50)
    
    # Load model at startup
    if not load_model():
        print("❌ Failed to load model. Exiting...")
        exit(1)
    
    print("\n✅ Server ready!")
    print("📍 Endpoint: http://127.0.0.1:9000/inference")
    print("🌍 Listening on: 0.0.0.0:9000")
    print("💡 Optimized for Arabic pronunciation detection")
    print("\n🛑 Press Ctrl+C to stop\n")
    print("="*50)
    
    # Start Flask server
    app.run(host='0.0.0.0', port=9000, debug=False, threaded=True)
