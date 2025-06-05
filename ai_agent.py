import os
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

# Load environment variables
load_dotenv()

# Step 1: Setup API Keys for Groq, OpenRouter.ai, and Tavily
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # Your OpenRouter API key

# Step 2: Setup LLM & Tools
groq_llm = ChatGroq(model="llama-3.3-70b-versatile")
search_tool = TavilySearchResults(max_results=2)

system_prompt = "Act as an AI chatbot who is smart and friendly"

def get_response_from_openrouter(query, system_prompt):
    try:
        url = "https://api.openrouter.ai/v1/complete"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "openrouter-llama-3.3-70b",
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
            "max_tokens": 100
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an error if the response is not successful
        response_data = response.json()

        # Extract and return the AI message content
        ai_message = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return ai_message
    except requests.exceptions.RequestException as e:
        print(f"Error with OpenRouter API: {e}")
        return "Error communicating with OpenRouter API."

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider == "Groq":
        llm = ChatGroq(model=llm_id)
    elif provider == "OpenRouter":
        ai_message = get_response_from_openrouter(query, system_prompt)
        return ai_message
    
    # Search tool functionality
    tools = [TavilySearchResults(max_results=2)] if allow_search else []
    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt
    )

    # Interact with the agent
    state = {"messages": query}
    response = agent.invoke(state)
    messages = response.get("messages")
    ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]
    return ai_messages[-1]
