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

@app.route('/analyze-youtube', methods=['POST'])
def analyze_youtube_video():
    global current_video, video_analysis
    
    data = request.json
    youtube_url = data.get('youtube_url', '')
    
    if not youtube_url:
        return jsonify({'error': 'No YouTube URL provided'}), 400
    
    try:
        # Create uploads folder if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Download YouTube video with more robust error handling and retries
        try:
            # Extract video ID from URL
            video_id = None
            if 'youtube.com/watch?v=' in youtube_url:
                video_id = youtube_url.split('watch?v=')[1].split('&')[0]
            elif 'youtu.be/' in youtube_url:
                video_id = youtube_url.split('youtu.be/')[1].split('?')[0]
            
            if not video_id:
                return jsonify({'error': 'Could not extract video ID from URL'}), 400
            
            # Create a YouTube object with the video ID
            yt = YouTube(f'https://www.youtube.com/watch?v={video_id}')
            
            # Add a timeout and retry mechanism
            max_retries = 3
            retry_count = 0
            video_stream = None
            
            while retry_count < max_retries:
                try:
                    # Try to get the video stream
                    video_stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
                    if video_stream:
                        break
                except Exception as e:
                    print(f"Retry {retry_count + 1}/{max_retries}: {str(e)}")
                    retry_count += 1
                    time.sleep(1)  # Wait before retrying
            
            if not video_stream:
                return jsonify({'error': 'Could not find a suitable video stream after multiple attempts'}), 400
            
            # Generate a filename based on the video title or ID if title is not available
            video_title = secure_filename(yt.title) if yt.title else video_id
            filename = f"{video_title}_{video_stream.resolution}.mp4"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Check if the file already exists (to avoid re-downloading)
            if not os.path.exists(filepath):
                # Download the video with retries
                retry_count = 0
                download_success = False
                
                while retry_count < max_retries and not download_success:
                    try:
                        video_stream.download(output_path=app.config['UPLOAD_FOLDER'], filename=filename)
                        download_success = True
                    except Exception as e:
                        print(f"Download retry {retry_count + 1}/{max_retries}: {str(e)}")
                        retry_count += 1
                        time.sleep(2)  # Wait before retrying
                
                if not download_success:
                    return jsonify({'error': 'Failed to download the video after multiple attempts'}), 500
            
            # Process the video with Gemini
            try:
                # Analyze video
                analysis = analyze_video_with_gemini(filepath)
                
                # Store analysis in vector database
                store_analysis_in_vector_db(filename, analysis)
                
                # Update current video information
                current_video = filepath
                video_analysis = analysis
                
                # Add video metadata
                video_metadata = {
                    'title': yt.title,
                    'author': yt.author,
                    'length': yt.length,
                    'views': yt.views,
                    'thumbnail_url': yt.thumbnail_url,
                    'source': 'youtube',
                    'source_url': youtube_url
                }
                
                return jsonify({
                    'success': True,
                    'filename': filename,
                    'analysis': analysis,
                    'metadata': video_metadata,
                    'message': 'YouTube video downloaded and analyzed successfully'
                })
                
            except Exception as e:
                return jsonify({'error': f'Error processing video: {str(e)}'}), 500
                
        except Exception as e:
            return jsonify({'error': f'Error with YouTube video: {str(e)}'}), 400
    
    except Exception as e:
        return jsonify({'error': f'Error downloading YouTube video: {str(e)}'}), 500

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
        
        # Extract relevant contexts and timestamps from query results
        contexts = []
        timestamps = []
        
        if query_results and 'documents' in query_results and query_results['documents']:
            for i, doc in enumerate(query_results['documents'][0]):
                contexts.append(doc)
                
                # Extract timestamp from metadata if available
                if 'metadatas' in query_results and query_results['metadatas'][0]:
                    timestamp = query_results['metadatas'][0][i].get('timestamp', '')
                    if timestamp:
                        timestamps.append(timestamp)
        
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
