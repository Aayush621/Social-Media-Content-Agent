# Social Media Content Generator 🚀

A powerful AI-powered application that generates optimized content for multiple social media platforms simultaneously. The system uses advanced language models to create platform-specific content while maintaining brand consistency and generating complementary images.

![WhatsApp Image 2024-12-06 at 22 03 44](https://github.com/user-attachments/assets/19a57700-48ce-44f6-91a8-69dff392e150)

## Features ✨

- **Multi-Platform Support**: Generate content for:
  - Twitter
  - LinkedIn
  - Instagram
  - Blog posts

- **Smart Content Generation** 🤖

![WhatsApp Image 2024-12-06 at 22 03 04](https://github.com/user-attachments/assets/2988ecf3-6d6c-4dcf-88db-66a6a9819653)

  - Platform-specific formatting and tone
  - Hashtag suggestions
  - Call-to-action (CTA) recommendations
  - SEO optimization for blog posts

![WhatsApp Image 2024-12-06 at 22 02 32](https://github.com/user-attachments/assets/6ed4a6ae-1ad7-427f-866f-3509b1319120)

- **Image Generation** 🎨

![WhatsApp Image 2024-12-06 at 22 01 42](https://github.com/user-attachments/assets/c0d9851c-1a08-4557-a829-b402d6896a4e)

  - AI-powered image creation
  - Platform-optimized visuals
  - Downloadable high-quality images

![WhatsApp Image 2024-12-06 at 22 01 07](https://github.com/user-attachments/assets/dacc061d-a9af-4a34-a949-fda5cd299e92)

- **Brand Consistency** 🎯
  - Maintains brand voice across platforms
  - Customizable tone and style
  - Target audience consideration

![WhatsApp Image 2024-12-06 at 22 00 23](https://github.com/user-attachments/assets/bbaf8afa-18d6-48ca-adc3-3e33a473eed7)

## Prerequisites 📋

- Python 3.8+
- API keys for:
  - Google AI (Gemini)
  - Tavily
  - Hugging Face
  - LangChain

## Installation 🛠️

1. Clone the repository:
```bash
git clone <repository-url>
cd social-media-content-generator
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your API keys:
```env
LANGCHAIN_API_KEY=your_langchain_api_key
TAVILY_API_KEY=your_tavily_api_key
GOOGLE_API_KEY=your_google_api_key
HUGGINGFACE_TOKEN=your_huggingface_token
```

## Running the Application 🚀

1. Start the FastAPI backend:
```bash
uvicorn api:app --reload
```

2. Open the frontend:
- Navigate to the directory containing `index.html`
- Open `index.html` in a web browser or use a local server:
```bash
python -m http.server 5500
```

3. Access the application at:
- Frontend: `http://localhost:5500`
- API Documentation: `http://localhost:8000/docs`

## Project Structure 📁

![content_generation_graph](https://github.com/user-attachments/assets/4db68848-24b9-4d8a-8e17-51f489e0a62c)


```
social-media-content-generator/
├── api.py              # FastAPI backend
├── agent.py            # AI agent logic and graph setup
├── prompts.py          # AI prompt templates
├── index.html          # Frontend interface
└── requirements.txt    # Python dependencies
```

## API Endpoints 🔌

- `POST /generate-content`
  - Generates content for specified platforms
  - Accepts text and platform preferences
  - Returns generated content and images

- `GET /supported-platforms`
  - Returns list of supported social media platforms

## Usage Example 💡

1. Open the web interface
2. Fill in your brand details:
   - Business name
   - Industry
   - Target audience
   - Brand tone
3. Enter your social media handles
4. Write your content text
5. Select target platforms
6. Choose image style preferences
7. Click "Generate Content"
8. Review and download generated content and images

## Error Handling 🔧

The application includes comprehensive error handling:
- Input validation
- API error responses
- Generation timeout handling
- Network error management

## Contributing 🤝

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
