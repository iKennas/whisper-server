#!/usr/bin/env python3
"""
Faster-Whisper Server for Arabic Speech Recognition
Optimized for real-time language learning pronunciation detection
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel
import tempfile
import os
import logging
import io

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model - load once at startup
model = None

def load_model():
    """Load Faster-Whisper model"""
    global model
    try:
        logger.info("Loading Faster-Whisper model (base)...")
        # Use base model for good balance of speed and accuracy
        # device="cpu" for CPU, device="cuda" for GPU
        # compute_type="int8" for faster inference on CPU
        model = WhisperModel("base", device="cpu", compute_type="int8")
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
        with tempfile.NamedTemporaryFile(delete=False, suffix='.flac') as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Transcribe with Faster-Whisper (WORD-LEVEL DATA)
            # beam_size: higher = more accurate but slower (1-5 recommended)
            # best_of: number of candidates (1-5)
            # temperature: controls randomness (0.0 = deterministic)
            segments, info = model.transcribe(
                tmp_path, 
                language=language,
                beam_size=5,  # Higher for better accuracy
                best_of=2,    # Get 2 candidates for better word confidence
                temperature=0.0,  # Deterministic for pronunciation checking
                vad_filter=True,  # Filter out silence
                word_timestamps=True  # ✅ ENABLE word-level timing & confidence
            )
            
            # Extract word-level data with timestamps and probabilities
            # IMPORTANT: segments is a generator, must convert to list first
            segments_list = list(segments)
            
            words_data = []
            full_text_parts = []
            
            logger.info(f"   Processing {len(segments_list)} segment(s)...")
            
            for segment in segments_list:
                segment_text = segment.text.strip()
                logger.info(f"   Segment text: '{segment_text}'")
                
                # Check if segment has words attribute
                has_words = hasattr(segment, 'words')
                logger.info(f"   - Has 'words' attribute: {has_words}")
                
                if has_words and segment.words is not None:
                    try:
                        # Try to convert words to list (might be generator or None)
                        words_list = list(segment.words)
                        logger.info(f"   - Found {len(words_list)} word(s) in segment")
                        
                        if words_list:
                            for word_info in words_list:
                                word_text = word_info.word.strip()
                                if word_text:  # Skip empty words
                                    word_data = {
                                        "word": word_text,
                                        "start": round(word_info.start, 2),
                                        "end": round(word_info.end, 2),
                                        "probability": round(word_info.probability, 3)
                                    }
                                    words_data.append(word_data)
                                    full_text_parts.append(word_text)
                                    logger.info(f"     * '{word_text}': {word_info.probability:.3f}")
                        else:
                            logger.warning(f"   - Words list is empty for segment")
                            full_text_parts.append(segment_text)
                    except Exception as e:
                        logger.error(f"   - Error extracting words: {e}")
                        full_text_parts.append(segment_text)
                else:
                    # Fallback if no word-level data
                    logger.warning(f"   - No word-level data for segment")
                    if segment_text:
                        full_text_parts.append(segment_text)
            
            # Combine full text
            full_text = " ".join(full_text_parts)
            
            logger.info(f"✅ Transcription: \"{full_text}\"")
            logger.info(f"   Language: {info.language} (prob: {info.language_probability:.2f})")
            logger.info(f"   Total words with timestamps: {len(words_data)}")
            
            # ⚠️ FALLBACK: If no word-level data, create estimated words from text
            if not words_data and full_text:
                logger.warning("⚠️ No word-level timestamps available, creating fallback words...")
                words = full_text.split()
                estimated_duration = 0.5  # Estimate 0.5 seconds per word
                for i, word in enumerate(words):
                    if word.strip():
                        words_data.append({
                            "word": word.strip(),
                            "start": round(i * estimated_duration, 2),
                            "end": round((i + 1) * estimated_duration, 2),
                            "probability": 0.75  # Default confidence for fallback
                        })
                logger.info(f"   Created {len(words_data)} fallback word(s)")
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            return jsonify({
                "text": full_text,
                "language": info.language,
                "language_probability": info.language_probability,
                "words": words_data,  # ✅ Word-level data with timestamps & confidence
                "total_duration": words_data[-1]["end"] if words_data else 0.0
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
        "model_name": "faster-whisper-base"
    })

if __name__ == '__main__':
    print("="*50)
    print("🎤 Faster-Whisper Server for Arabic Learning")
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

