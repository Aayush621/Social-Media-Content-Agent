from typing_extensions import TypedDict, List, Literal
from pydantic import BaseModel, Field
from langgraph.graph.message import MessagesState
import operator
from typing import Annotated, List, Dict
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Send
from langchain_google_genai import ChatGoogleGenerativeAI
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import re
import os

load_dotenv()

def _set_env(name: str):
    env_value = os.getenv(name)
    if not env_value:
        raise ValueError(f"Missing environment variable: {name}. Please check your .env file.")
    os.environ[name] = env_value

_set_env("LANGCHAIN_API_KEY")
_set_env("TAVILY_API_KEY")
_set_env("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=30,
    max_retries=5,
    retry_wait_time=2
    # other params...
)
user_details = {
  "user_name": "CalveinKlein Design Team",
  "business_name": "Calvein Klein",
  "industry": "Fashion",
  "business_type": "Clothing Startup",
  "target_audience": ["Young People", "Fashion Enthusiasts", "GenZ Kids"],
  "tone": "Aesthetic",
  "objectives": ["Awareness", "Marketing"],
  "platforms": ["LinkedIn", "Twitter", "Instagram"],
  "preferred_platforms": ["Instagram", "Twitter"],
  "platform_specific_details": {
    "instagram_handle": "@ck_design",
    "twitter_handle": "@CalveinKlein",
    "linkedin_page": "linkedin.com/company/Ck",
    "medium_page": "medium.com/ck"
  },
  "campaigns": [
    {
      "title": "Get you desired clothes",
      "date": "2024-05-20",
      "platform": "Instagram",
      "success_metric": "1000+ Shares"
    }
  ],
  "popular_hashtags": ["#TrendingClothes", "#Fashion", "#Fashionverse"],
  "themes": ["Fashion", "Trending Clothes"],
  "short_length": 280,
  "long_length": 2000,
  "assets_link": "https://drive.google.com/drive/folders/langgraph-assets",
  "colors": ["#1E88E5", "#FFC107"],
  "brand_keywords": ["Innovative", "Efficient"],
  "restricted_keywords": ["Buggy", "Outdated"],
  "competitors": ["HnM", "Levi's"],
  "competitor_metrics": ["Content Shares", "Follower Growth"],
  "posting_schedule": ["Tuesday 10 AM", "Friday 3 PM"],
  "formats": ["Carousel", "Quote"],
  "personal_preferences": "Use technical terms but keep explanations concise."
}

Platform = Literal["Twitter", "LinkedIn", "Instagram", "Blog"]

class InputState(TypedDict):
    text: str
    platforms: list[Platform]

class SumamryOutputState(TypedDict):
    text: str
    text_summary: str
    platforms: list[Platform]

class ResearchOutputState(TypedDict):
    text: str
    research: str
    platforms: list[Platform]

class IntentMatchingInputState(TypedDict):
    text: str
    research: str
    platforms: list[Platform]
    content_strategy: dict

class ContentStrategyState(TypedDict):
    text: str
    research: str
    platforms: list[Platform]
    content_strategy: dict  # Will contain platform-specific strategies

class FinalState(TypedDict):
    contents: Annotated[list, operator.add]

class GeneratedContent(TypedDict):
    generated_content: str

summ_model = llm

model = llm

sumamry_prompt = ChatPromptTemplate.from_template("""
Taks: You need to give a summary of this given text. This summary will help the user to get the idea of the whole text. Do not miss anything important as this summary will take place in Research.

Text:
 {text}

""")

research_agent_prompt = ChatPromptTemplate.from_template("""
You are a member of the Content Generation Team. Your primary task is to research and analyze the provided details to enhance the content creation process.

Here are the client's details:
{user_details}

Below is the summary of the content for which the client wants to generate textual material:
{text_summary}

Also

The client wants to create content for the following platforms:
{platforms}

Your task is to focus on content development enhancements :

- Suggest best keywords or hashtags relevant to the platform and the content intent.
- Identify key points or themes that should be highlighted or have been emphasized in previous posts.
- Propose possible content elements or formats (e.g., lists, visuals, tone adjustments) tailored to the platform's audience and characteristics.
- .... Anything which is enhances content


Response Format:
[
post1",
 post2",...
]
""")

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
{{
    "platform_strategies": {{
        [platforms]: {{
            "content_type": "",
            "format": "",
            "key_messages": [],
            "tone": "",
            "specific_elements": []
        }}
    }}
}}

Generate entries only for the platforms specified in the Target Platforms list.
""")

research_tool = TavilySearchResults(
    max_results=2,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)

class ReserachQuestions(BaseModel):
    questions: List[str] = Field(..., description="A list of questions that need to be researched.")

def summary_text(state: InputState) -> SumamryOutputState:
    print("******* Generating summary of the given text *************")
    summary = summ_model.invoke(state["text"]).content
    return {"text": state["text"], "platforms": state["platforms"], "text_summary": summary}

def research_node(state: SumamryOutputState) -> ResearchOutputState:
    print("******* Researching for the best content *************")
    input_ = {"user_details": user_details, "text_summary": state["text_summary"], "platforms": state["platforms"]}
    res = model.with_structured_output(ReserachQuestions).invoke(research_agent_prompt.invoke(input_))
    # Access questions using dot notation instead of dictionary syntax
    response = research_tool.batch(res.questions)
    research = ""
    for i, ques in enumerate(res.questions):
        research += "question: " + ques + "\n"
        research += "Answers" + "\n\n".join([res["content"] for res in response[i]]) + "\n\n"

    return {"text": state["text"], "platforms": state["platforms"], "research": research}

def content_strategy_node(state: ResearchOutputState) -> ContentStrategyState:
    print("******* Generating Content Strategy *************")
    input_ = {
        "user_details": user_details,
        "text_summary": state["text"],
        "research": state["research"],
        "platforms": state["platforms"]
    }

    # Get the strategy as a string
    strategy_response = model.invoke(content_strategy_prompt.invoke(input_))

    # Parse the JSON string into a dictionary
    import json
    try:
        strategy_dict = json.loads(strategy_response.content)
    except json.JSONDecodeError:
        # Fallback in case the response isn't valid JSON
        strategy_dict = {"platform_strategies": {}}

    return {
        "text": state["text"],
        "research": state["research"],
        "platforms": state["platforms"],
        "content_strategy": strategy_dict
    }

def IntentMatching(state: ResearchOutputState) -> Dict:
    print("******* Sending data to each Platform *************")
    # Return a dictionary with the required state information
    return {
        "text": state["text"],
        "research": state["research"],
        "content_strategy": state.get("content_strategy", {}),
        "platforms": state["platforms"]
    }

instagram_prompt = ChatPromptTemplate.from_template("""
You are a creative social media strategist specializing in Instagram content.

**Input Details:**
1. Text: {text}
2. Research: {research}
3. Content Strategy:
   - Content Type: {{content_strategy.get('content_type', 'engaging and visual')}}
   - Format: {{content_strategy.get('format', 'single post')}}
   - Key Messages: {{content_strategy.get('key_messages', ['brand values', 'product features'])}}
   - Tone: {{content_strategy.get('tone', 'professional yet approachable')}}
4. Current: {{post_number}}

Create an Instagram post following these guidelines and the research provided.

Your task is to create an **Instagram post caption** and provide the following:
- **Engaging Caption**: Write a compelling caption that aligns with the given text, highlights the key points, and uses an **inspirational or engaging tone** (as per the audience).
- **Hashtag Suggestions**: Suggest at least 10 hashtags that are **trending and relevant** to the content and target audience.
- **Call-to-Action (CTA)**: Include a specific action to encourage user engagement (e.g., comment, tag friends, visit website).
- **Emoji Usage**: Add appropriate emojis to make the caption lively and engaging, without overdoing it.

**Special Guidelines:**
1. Analyse the number of posts to be generated through user's query.
2. Keep the caption within 2200 characters but aim for 150–300 characters for better engagement.
3. Ensure hashtags balance **broad reach (#FitnessGoals)** and **niche relevance (#EcoFitFashion)**.
4. Optimize for Instagram's algorithm by starting with a **hook** (e.g., a question or statement).

**Response Format:**
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
   - Content Type: {{content_strategy.get('content_type', 'engaging and visual')}}
   - Format: {{content_strategy.get('format', 'single post')}}
   - Key Messages: {{content_strategy.get('key_messages', ['brand values', 'product features'])}}
   - Tone: {{content_strategy.get('tone', 'professional yet approachable')}}
4. Current: {{post_number}}

Your task is to create **Twitter content** with the following specifications:
- **Tweet**: Craft a tweet that conveys the essence of the text in **280 characters or less**, ensuring clarity, conciseness, and a conversational tone.
- **Hashtag Suggestions**: Include up to 3 hashtags that enhance visibility and are platform-specific.
- **Thread**: If the content cannot fit in a single tweet, create a **thread** with concise, numbered tweets that maintain flow and engagement.

**Special Guidelines:**
1. Analyse the number of posts to be generated through user's query.
2. Start with a **strong hook** in the first tweet to grab attention.
3. Use one or two relevant keywords or phrases identified in the research.
4. Maintain a balance between **professional** and **relatable** language.

**Response Format:**
Tweet: [Your tweet here]
Hashtags: [#hashtag1, #hashtag2, ...]
Thread:
1. [First tweet in the thread]
2. [Second tweet in the thread]
...

""")

linkedin_prompt = ChatPromptTemplate.from_template("""
You are a professional LinkedIn content creator, focused on crafting posts that establish thought leadership and build connections.

**Input Details:**
1. Text: {text}
2. Research: {research}
3. Content Strategy:
   - Content Type: {{content_strategy.get('content_type', 'engaging and visual')}}
   - Format: {{content_strategy.get('format', 'single post')}}
   - Key Messages: {{content_strategy.get('key_messages', ['brand values', 'product features'])}}
   - Tone: {{content_strategy.get('tone', 'professional yet approachable')}}

Your task is to create a **LinkedIn post** with the following details:
- **Post Content**: Write a professional, thoughtful post elaborating on the text, tailored to LinkedIn’s audience. Highlight the key takeaways or updates and use a **formal yet engaging tone**.
- **Hashtags**: Suggest up to 5 hashtags relevant to LinkedIn’s professional audience.
- **CTA**: Include a CTA encouraging engagement (e.g., “Share your thoughts,” “Let us know how you tackle this,” or “Visit our page for more”).

**Special Guidelines:**
1. Analyse the number of posts to be generated through user's query.
2. Aim for **150–300 words**, focusing on storytelling and professional insights.
3. Structure the post with:
   - A **hook** to grab attention.
   - The main body with value-driven insights.
   - A concluding CTA.
4. Avoid using jargon unless contextually relevant.
5. Ensure hashtags are business-focused and professional.

**Response Format:**
Post: [Your LinkedIn post here]
Hashtags: [#hashtag1, #hashtag2, ...]
CTA: [Call-to-Action here]

""")

blog_prompt = ChatPromptTemplate.from_template("""
You are a content writer specializing in blogs that captivate readers and provide actionable insights.

**Input Details:**
1. Text: {text}
2. Research: {research}
3. Content Strategy:
   - Content Type: {{content_strategy.get('content_type', 'engaging and visual')}}
   - Format: {{content_strategy.get('format', 'single post')}}
   - Key Messages: {{content_strategy.get('key_messages', ['brand values', 'product features'])}}
   - Tone: {{content_strategy.get('tone', 'professional yet approachable')}}
Your task is to create a **markdown-formatted blog post** with the following structure:
- **Title**: Create an eye-catching and SEO-friendly blog title.
- **Introduction**: Write an engaging opening paragraph that sets the context and hooks the reader.
- **Main Body**: Elaborate on the text using the research to provide insights, examples, and supporting details. Structure it into sections with headings (H2/H3).
- **Conclusion**: Summarize key takeaways and include a CTA encouraging readers to take the next step.

**Special Guidelines:**
1. Analyse the number of posts to be generated through user's query.
2. Use a tone aligned with the target audience (e.g., casual for general readers, formal for professionals).
3. Optimize for SEO by incorporating keywords from the research naturally into the content.
4. Ensure readability by using bullet points, numbered lists, and short paragraphs.
5. Keep the blog **800–1500 words**.

**Response Format:**
```markdown
# [Title of the Blog]

## Introduction
[Your introduction here]

## Section 1: [Heading]
[Content]

## Section 2: [Heading]
[Content]

## Conclusion
[Conclusion with CTA]

""")

def Insta(state: IntentMatchingInputState) -> FinalState:
    if not "Instagram" in state["platforms"]:
        return {"contents": [""]}

    # Extract number of posts from the text using a simple parsing
    import re
    num_posts = 1  # default
    match = re.search(r'(\d+)\s+posts?\s+for\s+Instagram', state["text"], re.IGNORECASE)
    if match:
        num_posts = int(match.group(1))

    content_strategy = state.get("content_strategy", {}).get("platform_strategies", {}).get("Instagram", {})

    all_posts = []
    for i in range(num_posts):
        prompt_input = {
            "text": state["text"],
            "research": state["research"],
            "content_strategy": content_strategy,
            "post_number": f"Post {i+1} of {num_posts}"
        }
        res = model.invoke(instagram_prompt.invoke(prompt_input))
        all_posts.append(f"Instagram Post #{i+1}:\n{res.content}")

    return {"contents": all_posts}

def Twitter(state: IntentMatchingInputState) -> FinalState:
    if not "Twitter" in state["platforms"]:
        return {"contents": [""]}

    # Extract number of posts
    num_posts = 1  # default
    match = re.search(r'(\d+)\s+posts?\s+for\s+Twitter', state["text"], re.IGNORECASE)
    if match:
        num_posts = int(match.group(1))

    content_strategy = state.get("content_strategy", {}).get("platform_strategies", {}).get("Twitter", {})

    all_posts = []
    for i in range(num_posts):
        prompt_input = {
            "text": state["text"],
            "research": state["research"],
            "content_strategy": content_strategy,
            "post_number": f"Post {i+1} of {num_posts}"
    }
        res = model.invoke(twitter_prompt.invoke(prompt_input))
        all_posts.append(f"Twitter Post #{i+1}:\n{res.content}")

    return {"contents": all_posts}

def Linkedin(state: IntentMatchingInputState) -> FinalState:
    if not "Linkedin" in state["platforms"]:
        return {"contents": [""]}

    # Get content strategy for LinkedIn or use defaults
    content_strategy = state.get("content_strategy", {}).get("platform_strategies", {}).get("Linkedin", {})

    prompt_input = {
        "text": state["text"],
        "research": state["research"],
        "content_strategy": content_strategy
    }
    res = model.invoke(linkedin_prompt.invoke(prompt_input))
    return {"contents": [res.content]}

def Blog(state: IntentMatchingInputState) -> FinalState:
    if not "Blog" in state["platforms"]:
        return {"contents": [""]}

    # Get content strategy for Blog or use defaults
    content_strategy = state.get("content_strategy", {}).get("platform_strategies", {}).get("Blog", {})

    prompt_input = {
        "text": state["text"],
        "research": state["research"],
        "content_strategy": content_strategy
    }
    res = model.invoke(blog_prompt.invoke(prompt_input))
    return {"contents": [res.content]}
def combining_content(state:FinalState) -> GeneratedContent:
    final_content = ""
    for content in state["contents"]:
        final_content += content + "\n\n"
    return {"generated_content": final_content}
def create_graph() :

    """ Create and return the content genration graph."""
    builder = StateGraph(input=InputState, output=GeneratedContent)

    # Nodes
    builder.add_node("summary_node",summary_text)
    builder.add_node("research_node", research_node)
    builder.add_node("intent_matching_node", IntentMatching)
    builder.add_node("instagram", Insta)
    builder.add_node("twitter", Twitter)
    builder.add_node("linkedin", Linkedin)
    builder.add_node("blog", Blog)
    builder.add_node("combine_content", combining_content)
    builder.add_node("content_strategy_node", content_strategy_node)


    # Flow
    builder.add_edge(START, "summary_node")
    builder.add_edge("summary_node", "research_node")

    # Update flow
    builder.add_edge("research_node", "content_strategy_node") # Make sure content_strategy_node runs before others dependent on its output
    builder.add_edge("content_strategy_node", "intent_matching_node")
    builder.add_edge("intent_matching_node", "instagram")
    builder.add_edge("intent_matching_node", "twitter")
    builder.add_edge("intent_matching_node", "linkedin")
    builder.add_edge("intent_matching_node", "blog")
    builder.add_edge("blog", "combine_content")
    builder.add_edge("twitter", "combine_content")
    builder.add_edge("instagram", "combine_content")
    builder.add_edge("linkedin", "combine_content")
    builder.add_edge("combine_content", END)

    return builder.compile()

graph = create_graph()

# res = graph.invoke({"text": """

# Calvin Klein is one of the world’s leading global fashion lifestyle brands with a history of bold, non-conformist ideals that inform everything we do. Founded in New York in 1968, the brand’s minimalist and sensual aesthetic drives our approach to product design and communication, creating a canvas that offers the possibility of limitless self-expression. The Calvin Klein brands – CK Calvin Klein, Calvin Klein, Calvin Klein Jeans, Calvin Klein Underwear, and Calvin Klein Performance – are connected by the intention and purpose of elevating everyday essentials to globally iconic status. Each of the brands has a distinct identity and position in the retail landscape, providing us the opportunity to market a range of universally appealing products to domestic and international consumers with a variety of needs. Our products are underpinned by responsible design, high-quality construction, and the elimination of all unnecessary details. We strive for unique and dimensional pieces that continuously wear well and remain relevant season after season. Global retail sales of Calvin Klein products were approximately $8.5 billion in 2021.

# Calvin Klein continues to solidify its position as an innovator of emerging digital platforms and modern marketing campaigns. PVH acquired Calvin Klein in 2003 and continues to oversee a focused approach to growing the brand’s worldwide relevance, presence, and long term growth.
# Generate 4 posts for Instagram and only 2 posts for Twitter for Winter Apparels
# """, "platforms": ["Instagram","Twitter"]})

# print(res["generated_content"])
