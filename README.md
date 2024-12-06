# Social Media Content Generator 🚀

A powerful AI-powered application that generates optimized content for multiple social media platforms simultaneously. The system uses advanced language models to create platform-specific content while maintaining brand consistency and generating complementary images.

![WhatsApp Image 2024-12-06 at 22 03 44](https://github.com/user-attachments/assets/c1acfd73-33a9-4ee3-bb5d-831574225954)


## Features ✨

- **Multi-Platform Support**: Generate content for:
  - Twitter
  - LinkedIn
  - Instagram
  - Blog posts

- **Smart Content Generation** 🤖

![WhatsApp Image 2024-12-06 at 22 03 04](https://github.com/user-attachments/assets/4d50b7a4-9ba4-4d61-bdea-3ad8b497aba2)


  - Platform-specific formatting and tone
  - Hashtag suggestions
  - Call-to-action (CTA) recommendations
  - SEO optimization for blog posts

![WhatsApp Image 2024-12-06 at 22 02 32](https://github.com/user-attachments/assets/6d82c734-38b6-474c-876e-e632d248fc0e)


- **Image Generation** 🎨

![WhatsApp Image 2024-12-06 at 22 01 42](https://github.com/user-attachments/assets/4340d374-975e-4ce0-97cd-06edb7e56908)


  - AI-powered image creation
  - Platform-optimized visuals
  - Downloadable high-quality images

![WhatsApp Image 2024-12-06 at 22 01 07](https://github.com/user-attachments/assets/bb50823d-52f4-46fc-965a-9beb392d42c2)


- **Brand Consistency** 🎯
  - Maintains brand voice across platforms
  - Customizable tone and style
  - Target audience consideration

![WhatsApp Image 2024-12-06 at 22 00 23](https://github.com/user-attachments/assets/60aed1e9-21b4-4cfd-9a70-9c190699817f)

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

![content_generation_graph](https://github.com/user-attachments/assets/64b9c98b-542b-45e7-a6e4-e080c6057942)


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
