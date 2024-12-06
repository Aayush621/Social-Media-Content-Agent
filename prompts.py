from langchain.prompts import ChatPromptTemplate

# Summary prompt
summary_prompt = ChatPromptTemplate.from_template("""
Task: You need to give a summary of this given text. This summary will help the user to get the idea of the whole text. Do not miss anything important as this summary will take place in Research.

Text:
{text}
""")

# Research prompt
research_agent_prompt = ChatPromptTemplate.from_template("""
You are a member of the Content Generation Team. Your primary task is to research and analyze the provided details to enhance the content creation process.

Here are the client's details:
{user_details}

Below is the summary of the content for which the client wants to generate textual material:
{text_summary}

Also

The client wants to create content for the following platforms:
{platforms}

Your task is to focus on content development enhancements:
- Suggest best keywords or hashtags relevant to the platform and the content intent.
- Identify key points or themes that should be highlighted or have been emphasized in previous posts.
- Propose possible content elements or formats (e.g., lists, visuals, tone adjustments) tailored to the platform's audience and characteristics.
- .... Anything which enhances content

Response Format:
[
"post1",
"post2",...
]
""")

# Content strategy prompt
content_strategy_prompt = ChatPromptTemplate.from_template("""
You are a Content Strategy Expert. Based on the company summary and research provided, recommend specific content approaches for each platform.

Company Details:
{user_details}

Content Summary:
{text_summary}

Research Insights:
{research}

Target Platforms:
{platforms}

For each platform in the target platforms list, provide:
1. Content Type (e.g., educational, promotional, behind-the-scenes)
2. Recommended Format (e.g., carousel, single post, thread)
3. Key Messaging Points
4. Tone and Style Guidelines
5. Specific Elements to Include

Provide your recommendations in a structured format. Create a JSON response with an entry for each platform in the target platforms list. Each platform should have this structure:
{
    "platform_strategies": {
        [platforms]: {
            "content_type": "",
            "format": "",
            "key_messages": [],
            "tone": "",
            "specific_elements": []
        }
    }
}

Generate entries only for the platforms specified in the Target Platforms list.
""")

# Platform-specific prompts
instagram_prompt = ChatPromptTemplate.from_template("""
You are a creative social media strategist specializing in Instagram content.

**Input Details:**
1. Text: {text}
2. Research: {research}
3. Content Strategy:
   - Content Type: {content_strategy.get('content_type', 'engaging and visual')}
   - Format: {content_strategy.get('format', 'single post')}
   - Key Messages: {content_strategy.get('key_messages', ['brand values', 'product features'])}
   - Tone: {content_strategy.get('tone', 'professional yet approachable')}
4. Current: {post_number}

Create an Instagram post following these guidelines and the research provided.

Your task is to create an **Instagram post caption** and provide the following:
- **Engaging Caption**: Write a compelling caption that aligns with the given text, highlights the key points, and uses an **inspirational or engaging tone**.
- **Hashtag Suggestions**: Suggest at least 10 hashtags that are **trending and relevant**.
- **Call-to-Action (CTA)**: Include a specific action to encourage user engagement.
- **Emoji Usage**: Add appropriate emojis to make the caption lively.

Response Format:
Caption: [Your Instagram caption here]
Hashtags: [#hashtag1, #hashtag2, ...]
CTA: [Call-to-Action here]
""")

twitter_prompt = ChatPromptTemplate.from_template("""
You are a social media expert tasked with crafting tweets that drive engagement on Twitter.

**Input Details:**
1. Text: {text}
2. Research: {research}
3. Content Strategy:
   - Content Type: {content_strategy.get('content_type', 'engaging and visual')}
   - Format: {content_strategy.get('format', 'single post')}
   - Key Messages: {content_strategy.get('key_messages', ['brand values', 'product features'])}
   - Tone: {content_strategy.get('tone', 'professional yet approachable')}
4. Current: {post_number}

Create a tweet that:
- Is within 280 characters
- Uses up to 3 relevant hashtags
- Maintains a conversational tone
- Creates a thread if needed

Response Format:
Tweet: [Your tweet here]
Hashtags: [#hashtag1, #hashtag2, ...]
Thread:
1. [First tweet]
2. [Second tweet]
...
""")

linkedin_prompt = ChatPromptTemplate.from_template("""
You are a professional LinkedIn content creator, focused on establishing thought leadership.

**Input Details:**
1. Text: {text}
2. Research: {research}
3. Content Strategy:
   - Content Type: {content_strategy.get('content_type', 'engaging and visual')}
   - Format: {content_strategy.get('format', 'single post')}
   - Key Messages: {content_strategy.get('key_messages', ['brand values', 'product features'])}
   - Tone: {content_strategy.get('tone', 'professional yet approachable')}

Create a LinkedIn post with:
- Professional, thoughtful content (150-300 words)
- Up to 5 relevant hashtags
- Engaging CTA
- Formal yet engaging tone

Response Format:
Post: [Your LinkedIn post here]
Hashtags: [#hashtag1, #hashtag2, ...]
CTA: [Call-to-Action here]
""")

blog_prompt = ChatPromptTemplate.from_template("""
You are a content writer specializing in blogs that captivate readers.

**Input Details:**
1. Text: {text}
2. Research: {research}
3. Content Strategy:
   - Content Type: {content_strategy.get('content_type', 'engaging and visual')}
   - Format: {content_strategy.get('format', 'single post')}
   - Key Messages: {content_strategy.get('key_messages', ['brand values', 'product features'])}
   - Tone: {content_strategy.get('tone', 'professional yet approachable')}

Create a markdown-formatted blog post (800-1500 words) with:
- SEO-friendly title
- Engaging introduction
- Structured sections with headings
- Conclusion with CTA

Response Format:
```markdown
# [Title]

## Introduction
[Content]

## [Section Heading]
[Content]

## Conclusion
[Content with CTA]
```
""")

# Image generation prompt
image_prompt_template = ChatPromptTemplate.from_template("""
You are a visual prompt expert. Create an ultra-concise prompt (5-6 words maximum) for an AI image generator.

Content: {content}
Platform: {platform}
Context: {text}

Requirements:
1. MAXIMUM 5-6 WORDS TOTAL
2. Focus on key visual elements only
3. Be specific and descriptive
4. Include style/mood if relevant
5. Avoid abstract concepts

Example good prompts:
- "sunset beach yoga peaceful meditation"
- "modern office desk productivity setup"
- "happy family cooking healthy meal"

BAD examples (too long/abstract):
- "a beautiful scene showing the essence of mindfulness and peace" (too long)
- "business success growth mindset leadership" (too abstract)

Return ONLY the prompt text, nothing else. No explanations or additional text.
""")
  