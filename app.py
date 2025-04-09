import os
import base64
import json
import tempfile
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import google.generativeai as genai
import chromadb
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize ChromaDB
chroma_client = chromadb.Client()
video_collection = chroma_client.create_collection(name="video_analysis")

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
        Analyze this video in detail and provide a structured output with:
        1. A timestamp-based breakdown (every few seconds)
        2. Detailed description of objects, people, and activities visible at each timestamp
        3. Any text visible in the video
        4. A transcript of any spoken content
        5. Overall summary of the video content
        
        Format the output as JSON with the following structure:
        {
            "timestamps": [
                {
                    "time": "00:00:05",
                    "objects": ["list of objects visible"],
                    "people": ["descriptions of people visible"],
                    "activities": ["descriptions of activities happening"],
                    "visible_text": "any text visible in the frame",
                    "transcript": "spoken content at this timestamp"
                }
            ],
            "summary": "overall summary of the video"
        }
        """
        
        # Call Gemini API with video content
        response = model.generate_content([
            prompt,
            {"mime_type": "video/mp4", "data": video_base64}
        ])
        
        # Extract and parse JSON from response
        # Note: We'll need to extract the JSON from the text response
        response_text = response.text
        
        # Find JSON content in the response (it might be wrapped in markdown code blocks)
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_content = response_text[json_start:json_end]
            analysis = json.loads(json_content)
        else:
            # If JSON parsing fails, use the raw text
            analysis = {"raw_analysis": response_text}
        
        return analysis
        
    except Exception as e:
        print(f"Error analyzing video: {str(e)}")
        return {"error": str(e)}

def store_analysis_in_vector_db(video_filename, analysis):
    """Store video analysis in ChromaDB vector database"""
    try:
        # Convert analysis to strings for storage
        timestamps = analysis.get("timestamps", [])
        
        # Store each timestamp as a separate document for more granular retrieval
        for i, timestamp in enumerate(timestamps):
            # Convert timestamp data to string
            timestamp_str = json.dumps(timestamp)
            
            # Create a unique ID for this timestamp entry
            doc_id = f"{video_filename}_{i}"
            
            # Store in vector database
            video_collection.add(
                documents=[timestamp_str],
                metadatas=[{
                    "video": video_filename,
                    "timestamp": timestamp.get("time", "unknown"),
                    "type": "timestamp_analysis"
                }],
                ids=[doc_id]
            )
        
        # Store the overall summary
        summary = analysis.get("summary", "")
        if summary:
            video_collection.add(
                documents=[summary],
                metadatas=[{
                    "video": video_filename,
                    "type": "summary"
                }],
                ids=[f"{video_filename}_summary"]
            )
            
        return True
    
    except Exception as e:
        print(f"Error storing in vector DB: {str(e)}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_video():
    global current_video, video_analysis
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the video with Gemini
        try:
            # Analyze video
            analysis = analyze_video_with_gemini(filepath)
            
            # Store analysis in vector database
            store_analysis_in_vector_db(filename, analysis)
            
            # Update current video information
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
    
    return jsonify({'error': 'Unknown error occurred'}), 500

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
        query_results = video_collection.query(
            query_texts=[user_message],
            n_results=5
        )
        
        # Extract relevant contexts from query results
        contexts = []
        if query_results and 'documents' in query_results and query_results['documents']:
            for doc in query_results['documents'][0]:
                contexts.append(doc)
        
        # Create a context string from the retrieved documents
        context_str = "\n".join(contexts)
        
        # Initialize Gemini model for chat
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Create prompt for Gemini
        prompt = f"""
        You are an assistant that can discuss the uploaded video based on its analysis.
        
        Video analysis information:
        {context_str}
        
        User question: {user_message}
        
        Please respond to the user's question about the video using only the information provided in the video analysis.
        """
        
        # Call Gemini API
        response = model.generate_content(prompt)
        
        return jsonify({
            'response': response.text
        })
        
    except Exception as e:
        return jsonify({'error': f'Error generating response: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
