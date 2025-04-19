import os
import base64
import json
import tempfile
import re
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import google.generativeai as genai
import chromadb
from dotenv import load_dotenv
from pytube import YouTube
import time
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Configure Google Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set. Please add it to your .env file or environment variables before starting the app.")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize SentenceTransformer embedding model (load once at startup)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize ChromaDB
chroma_client = chromadb.Client()
video_collection = chroma_client.get_or_create_collection(name="video_analysis")

# Store the current video information
current_video = None
video_analysis = None

def encode_video_to_base64(video_path):
    """Encode video file to base64 for Gemini API"""
    with open(video_path, "rb") as video_file:
        return base64.b64encode(video_file.read()).decode("utf-8")

def analyze_video_with_gemini(video_path):
    """Analyze video using Gemini 2.0 Flash model"""
    try:
        # Encode video to base64
        video_base64 = encode_video_to_base64(video_path)
        
        # Initialize Gemini model (using 2.0 Flash as specified)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Create prompt for detailed video analysis
        prompt = """
        Analyze this video in extreme detail and provide a comprehensive structured output with:
        
        1. A timestamp-based breakdown (every 5-10 seconds or at significant scene changes)
        2. For each timestamp, provide:
           - Detailed list of ALL objects visible (furniture, items, tools, etc.)
           - Detailed descriptions of ALL people visible (appearance, clothing, actions)
           - Comprehensive list of ALL activities happening
           - Any text visible in the frame (signs, screens, documents)
           - Transcript of any spoken content
           - Background elements (setting, environment, lighting)
           - Camera movements or angle changes
        3. Overall summary of the video content
        
        Format the output as JSON with the following structure:
        {
            "timestamps": [
                {
                    "time": "00:00:05",
                    "objects": ["detailed list of ALL objects visible"],
                    "people": ["detailed descriptions of ALL people visible"],
                    "activities": ["comprehensive descriptions of ALL activities happening"],
                    "visible_text": "any text visible in the frame",
                    "transcript": "spoken content at this timestamp",
                    "background": "description of setting and environment",
                    "camera": "description of camera movements or angles"
                }
            ],
            "summary": "detailed overall summary of the video"
        }
        
        It is CRITICAL that your response be valid JSON that can be parsed. Do not include any explanatory text before or after the JSON.
        Ensure you provide as much detail as possible for each category at each timestamp.
        """
        
        # Call Gemini API with video content
        response = model.generate_content([
            prompt,
            {"mime_type": "video/mp4", "data": video_base64}
        ])
        
        # Extract and parse JSON from response
        response_text = response.text
        
        # Find JSON content in the response (it might be wrapped in markdown code blocks)
        # First try to extract JSON from code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        
        if json_match:
            json_content = json_match.group(1)
            try:
                analysis = json.loads(json_content)
                return analysis
            except json.JSONDecodeError:
                pass  # If this fails, we'll try the next method
        
        # If no code blocks or parsing failed, try to find JSON directly
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_content = response_text[json_start:json_end]
            try:
                analysis = json.loads(json_content)
                return analysis
            except json.JSONDecodeError:
                # If JSON parsing fails, use the raw text
                return {"raw_analysis": response_text}
        else:
            # If no JSON found, use the raw text
            return {"raw_analysis": response_text}
        
    except Exception as e:
        print(f"Error analyzing video: {str(e)}")
        return {"error": str(e)}

def store_analysis_in_vector_db(video_filename, analysis):
    """Store video analysis in ChromaDB vector database with semantic embeddings"""
    try:
        timestamps = analysis.get("timestamps", [])

        for i, timestamp in enumerate(timestamps):
            text_parts = []
            # Combine all relevant fields for context-rich embedding
            for field in ["description", "activities", "objects", "people", "text", "transcript", "visible_text", "background", "camera"]:
                val = timestamp.get(field)
                if isinstance(val, list):
                    text_parts.append("; ".join(map(str, val)))
                elif val:
                    text_parts.append(str(val))
            # Add the time itself for reference
            if timestamp.get("time"):
                text_parts.append(f"Time: {timestamp['time']}")
            # Add start_time/end_time if present
            if timestamp.get("start_time"):
                text_parts.append(f"Start: {timestamp['start_time']}")
            if timestamp.get("end_time"):
                text_parts.append(f"End: {timestamp['end_time']}")
            chunk_text = " | ".join([x for x in text_parts if x])

            # DEBUG: Print what is being embedded
            print(f"[DEBUG] Storing chunk for {video_filename} at index {i}:")
            print(chunk_text)

            # Generate semantic embedding
            embedding = embedding_model.encode(chunk_text, show_progress_bar=False, normalize_embeddings=True).tolist()

            # Prepare metadata for ChromaDB, ensuring all values are str/int/float/bool (no None)
            def sanitize(val):
                return val if val is not None else ""
            metadata = {
                "video_filename": sanitize(video_filename),
                "timestamp_index": i,
                "start_time": sanitize(timestamp.get("start_time")),
                "end_time": sanitize(timestamp.get("end_time")),
                "time": sanitize(timestamp.get("time")),
                # Optionally, you can exclude 'raw' if it contains nested None values or is not needed for search
            }

            video_collection.add(
                documents=[chunk_text],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[f"{video_filename}_{i}"]
            )
    except Exception as e:
        print(f"Error storing analysis in vector DB: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_video():
    global current_video, video_analysis
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
        except Exception as e:
            return jsonify({'error': f'Could not save file: {str(e)}'}), 500
        
        # Process the video with Gemini
        try:
            analysis = analyze_video_with_gemini(filepath)
            if not analysis or (isinstance(analysis, dict) and analysis.get('error')):
                error_msg = analysis.get('error', 'Unknown analysis error') if isinstance(analysis, dict) else 'Unknown analysis error'
                return jsonify({'error': f'Video analysis failed: {error_msg}'}), 500
            store_analysis_in_vector_db(filename, analysis)
            current_video = filepath
            video_analysis = analysis
            return jsonify({
                'success': True,
                'filename': filename,
                'analysis': analysis,
                'message': 'Video uploaded and analyzed successfully'
            })
        except Exception as e:
            return jsonify({'error': f'Error processing video: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat_with_video():
    global current_video, video_analysis
    
    if not current_video:
        return jsonify({'error': 'No video has been uploaded yet'}), 400
    
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        # Search the vector database for relevant information
        print(f"[DEBUG] User query: {user_message}")
        query_results = video_collection.query(
            query_texts=[user_message],
            n_results=5
        )
        print(f"[DEBUG] ChromaDB query results: {query_results}")
        
        # Extract relevant contexts and timestamps from query results
        contexts = []
        timestamps = []
        
        if query_results and 'documents' in query_results and query_results['documents']:
            for i, doc in enumerate(query_results['documents'][0]):
                contexts.append(doc)
                
                # Extract start_time and end_time from metadata if available
                if 'metadatas' in query_results and query_results['metadatas'][0]:
                    meta = query_results['metadatas'][0][i]
                    start_time = meta.get('start_time', '')
                    end_time = meta.get('end_time', '')
                    # Format as [HH:MM:SS] or [start - end]
                    if start_time and end_time:
                        timestamps.append(f"[{start_time} - {end_time}]")
                    elif start_time:
                        timestamps.append(f"[{start_time}]")
        
        # Create a context string from the retrieved documents
        context_str = "\n".join(contexts)
        
        # Initialize Gemini model for chat
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Create prompt for Gemini with instruction to include timestamps
        prompt = f"""
        You are an assistant that can discuss the uploaded video based on its analysis.
        
        Video analysis information:
        {context_str}
        
        User question: {user_message}
        
        Please respond to the user's question about the video using only the information provided in the video analysis.
        
        IMPORTANT: In your response, whenever you reference a specific moment in the video, include the timestamp in this format: [HH:MM:SS].
        These timestamps should be accurate and based on the timestamps in the video analysis information.
        """
        
        # Call Gemini API
        response = model.generate_content(prompt)
        
        # Process the response to make timestamps clickable
        processed_response = {
            'text': response.text,
            'timestamps': timestamps
        }
        
        return jsonify({
            'response': processed_response
        })
        
    except Exception as e:
        return jsonify({'error': f'Error generating response: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
