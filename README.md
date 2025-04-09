# Video Analysis and Chat Application

This application allows you to upload videos, analyze them using Google's Gemini 2.0 Flash model, and chat with the video content. The application extracts detailed information from videos including:

- Timestamp-based breakdown of the video
- Objects, people, and activities visible at each timestamp
- Text visible in the video
- Transcript of spoken content
- Overall summary of the video

All this information is stored in a vector database (ChromaDB) to enable semantic search and natural language querying of the video content.

## Features

- Video upload and processing
- Detailed video analysis using Google Gemini 2.0 Flash
- Vector database storage for efficient retrieval
- Interactive chat interface to ask questions about the video
- Beautiful and responsive UI

## Video Upload Limitations

- Maximum video file size: 100MB
- Supported formats: MP4, MOV, AVI, and other common video formats
- Recommended video length: 1-5 minutes for optimal performance
- Longer videos (up to 10 minutes) can be processed but may take significantly more time
- Video quality: Higher quality videos will result in better analysis

## API Key Requirements

This application requires a Google API key with access to the Gemini 2.0 Flash model:

1. Visit the [Google AI Studio](https://makersuite.google.com/app/apikey) to get your API key
2. Enable the Gemini API in your Google Cloud Console
3. Add your API key to the `.env` file as shown in the setup instructions
4. **IMPORTANT**: Never commit your API key to version control. The `.env` file is included in `.gitignore` for this reason.

## Setup Instructions

1. Ensure you have Python 3.8+ installed
2. Clone this repository:
   ```
   git clone https://github.com/akashmishra97/Chat_With_Videos_Using_AI.git
   cd Chat_With_Videos_Using_AI
   ```
3. Create a virtual environment and activate it:
   ```
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```
4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
5. Create a `.env` file in the root directory with your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```
6. Run the application:
   ```
   python app.py
   ```
7. Open your browser and navigate to `http://127.0.0.1:5000`

## Usage

1. Upload a video file (MP4 format recommended)
2. Wait for the analysis to complete (this may take a few minutes depending on the video length)
3. Review the detailed analysis with timestamps
4. Use the chat interface to ask questions about the video content

## Technologies Used

- Google Gemini 2.0 Flash for video analysis
- Flask for the web server
- ChromaDB for vector storage
- JavaScript for the frontend

## Security Notes

- This application is designed for local use or development environments
- For production deployment, additional security measures would be needed
- API keys are stored in environment variables and not hardcoded in the application
- The `.env` file is included in `.gitignore` to prevent accidentally committing API keys

## Notes

- Maximum video file size is limited to 100MB
- Video analysis may take several minutes depending on the length and complexity of the video
- The quality of the analysis depends on the clarity and content of the video

## License

MIT
