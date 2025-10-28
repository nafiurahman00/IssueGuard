# Secret Detection Tool

A comprehensive machine learning-powered secret detection system that combines regex patterns with a fine-tuned CodeBERT model to identify sensitive information (API keys, tokens, passwords, etc.) in text and code. The project includes a FastAPI backend server and a Chrome extension for GitHub integration.

## 🌟 Features

- **ML-Powered Detection**: Fine-tuned Microsoft CodeBERT model for accurate secret detection
- **Hybrid Approach**: Combines regex patterns with machine learning for comprehensive coverage
- **FastAPI Backend**: High-performance async API with caching and batch processing
- **Chrome Extension**: Real-time secret detection on GitHub with visual indicators
- **Intelligent Caching**: LRU cache for improved performance on repeated checks
- **Multi-Pattern Support**: Detects various secret types (API keys, tokens, credentials, etc.)
- **Real-time Analysis**: Debounced checking with visual feedback (Grammarly-style)

## 📁 Project Structure

```
Tool/
├── api/                          # FastAPI backend
│   ├── __init__.py
│   ├── app.py                    # FastAPI application factory
│   ├── config.py                 # Configuration settings
│   ├── model_manager.py          # ML model loading and inference
│   ├── models.py                 # Pydantic data models
│   ├── regex_manager.py          # Regex pattern management
│   ├── routes.py                 # API route handlers
│   ├── service.py                # Business logic layer
│   └── utils.py                  # Utility functions
├── models/                       # Pre-trained ML models
│   └── balanced/
│       └── microsoft_codebert-base_complete/
├── SBMBot/                       # Chrome extension
│   ├── background.js             # Service worker
│   ├── content.js                # Content script for GitHub
│   ├── manifest.json             # Extension manifest
│   └── popup.html                # Extension popup UI
├── test/                         # Test scripts
│   ├── client.py
│   ├── client_multiple.py
│   ├── inference.py
│   ├── run_inference.py
│   ├── single_inference.py
│   └── test_*.py
├── main.py                       # Server entry point
└── requirements_fastapi.txt      # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip
- Chrome/Chromium browser (for extension)
- CUDA-compatible GPU (optional, for faster inference)

### Installation

1. **Clone the repository**
   ```powershell
   git clone <repository-url>
   cd Tool
   ```

2. **Install Python dependencies**
   ```powershell
   pip install -r requirements_fastapi.txt
   ```

3. **Verify model files**
   Ensure the CodeBERT model is present in `models/balanced/microsoft_codebert-base_complete/`

### Running the Backend Server

**Basic Usage:**
```powershell
python main.py
```

**Development Mode (with auto-reload):**
```powershell
python main.py --reload
```

**Custom Host and Port:**
```powershell
python main.py --host 0.0.0.0 --port 8080
```

**Production with Multiple Workers:**
```powershell
python main.py --workers 4
```

The server will start at `http://localhost:8000` by default.

### Installing the Chrome Extension

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top-right corner)
3. Click "Load unpacked"
4. Select the `SBMBot` folder from this project
5. The extension icon should appear in your Chrome toolbar

### Configuring the Extension

1. Make sure the backend server is running
2. Visit any GitHub repository page
3. The extension will automatically detect text areas and provide real-time feedback

## 📚 API Documentation

Once the server is running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### API Endpoints

#### `GET /`
Root endpoint with API information.

#### `GET /health`
Health check endpoint with service status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "patterns_loaded": 15,
  "device": "cuda"
}
```

#### `POST /detect`
Detect secrets in provided text.

**Request Body:**
```json
{
  "text": "AWS_SECRET_KEY=abc123def456...",
  "max_results": 100,
  "batch_size": 32
}
```

**Response:**
```json
{
  "secrets": [
    {
      "candidate_string": "abc123def456",
      "secret_type": "AWS Secret Key",
      "is_secret": true,
      "position_start": 15,
      "position_end": 27,
      "pattern_id": "aws_secret",
      "source": "ml_model",
      "from_cache": false
    }
  ],
  "all_candidates": [...],
  "processing_time_ms": 45.2,
  "cache_hit_rate": 0.0
}
```

## 🔧 Configuration

Configuration settings are managed in `api/config.py`. Key settings include:

- `HOST`: Server host (default: `127.0.0.1`)
- `PORT`: Server port (default: `8000`)
- `MODEL_PATH`: Path to the ML model
- `DEVICE`: Compute device (`cuda` or `cpu`)
- `MAX_WORKERS`: Thread pool size for inference
- `CACHE_SIZE`: LRU cache size

## 🧪 Testing

Run test scripts to verify functionality:

```powershell
# Test single inference
python test/single_inference.py

# Test API client
python test/test_client.py

# Test multiple requests
python test/client_multiple.py

# Test cache performance
python test/test_cache_client.py
```

## 🎯 Chrome Extension Features

- **Real-time Detection**: Automatic scanning as you type (with debouncing)
- **Visual Indicators**: Status indicator showing scan results
- **Inline Highlights**: Suspected secrets highlighted in text
- **Smart Tooltips**: Hover over highlights for detailed information
- **Non-intrusive**: Grammarly-style interface that doesn't disrupt workflow
- **GitHub Integration**: Works seamlessly with GitHub's text editors

## 🏗️ Architecture

### Backend Architecture

1. **FastAPI Application** (`app.py`): Handles HTTP requests with CORS support
2. **Service Layer** (`service.py`): Business logic with caching and async processing
3. **Model Manager** (`model_manager.py`): ML model loading and inference
4. **Regex Manager** (`regex_manager.py`): Pattern matching and candidate extraction
5. **Routes** (`routes.py`): API endpoint definitions

### Detection Pipeline

1. **Regex Extraction**: Fast pattern matching to find potential secrets
2. **Context Window**: Extract surrounding context for each candidate
3. **ML Classification**: CodeBERT model predicts if candidate is a real secret
4. **Cache Lookup**: Check LRU cache for previously analyzed candidates
5. **Result Aggregation**: Combine results from both methods

### Chrome Extension Architecture

- **Content Script** (`content.js`): Injects UI elements and monitors text areas
- **Background Worker** (`background.js`): Handles message passing
- **API Communication**: Sends requests to backend for analysis

## 🛡️ Security Considerations

- The tool is designed for **detection**, not prevention
- Always use environment variables and secret management tools in production
- Never commit real secrets to version control
- The Chrome extension sends text to localhost by default
- Consider network security when exposing the API

## 🔄 Performance Optimization

- **LRU Caching**: Reduces redundant ML inference calls
- **Batch Processing**: Processes multiple candidates efficiently
- **Async/Await**: Non-blocking I/O operations
- **Thread Pool**: Parallel inference execution
- **CUDA Support**: GPU acceleration when available

## 📝 Model Information

- **Base Model**: Microsoft CodeBERT (`microsoft/codebert-base`)
- **Training**: Fine-tuned on secret detection dataset
- **Format**: SafeTensors
- **Location**: `models/balanced/microsoft_codebert-base_complete/`

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🐛 Troubleshooting

### Server Won't Start
- Check if port 8000 is already in use
- Verify Python dependencies are installed
- Ensure model files are present and not corrupted

### Extension Not Working
- Verify the backend server is running
- Check browser console for errors (F12)
- Ensure the extension has proper permissions
- Try reloading the extension

### Poor Detection Accuracy
- Verify the ML model is loaded correctly
- Check regex patterns in `regex_manager.py`
- Review the confidence threshold settings
- Consider retraining the model with more data

### Performance Issues
- Enable CUDA if GPU is available
- Increase cache size in configuration
- Reduce batch size for lower memory usage
- Use production mode (disable `--reload`)

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review the API documentation at `/docs`

## 🙏 Acknowledgments

- Microsoft for the CodeBERT model
- FastAPI framework
- Transformers library by Hugging Face
- The open-source community

---

**Note**: This tool is for educational and security research purposes. Always follow responsible disclosure practices when finding real secrets.
