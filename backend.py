from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from ai_agent import get_response_from_ai_agent

# 1. Pydantic schema
class ChatRequest(BaseModel):
    model_name: str = Field(..., example="llama-3.3-70b-versatile")
    model_provider: str = Field(..., example="OpenRouter")   # "Groq" or "OpenRouter"
    system_prompt: str = Field(..., example="Act as an AI chatbot who is smart and friendly.")
    messages: List[str] = Field(..., example=["Hello! How are you?"])
    allow_search: bool = Field(..., example=False)

# 2. Allowed Models Configuration
ALLOWED_MODELS = {
    "Groq": {
        "llama3-70b-8192",
        "mixtral-8x7b-32768",
        "llama-3.3-70b-versatile",
    },
    "OpenRouter": {
        "openrouter-llama-3.3-70b",
        "mixtral-8x7b-32768",
    },
}

app = FastAPI(title="LangGraph AI Agent")

# 3. Endpoint for Chat
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    provider = request.model_provider

    # --- Validate provider -------------------------------------------------
    if provider not in ALLOWED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid provider. Choose from {', '.join(ALLOWED_MODELS.keys())}."
        )

    # --- Validate model ----------------------------------------------------
    if request.model_name not in ALLOWED_MODELS[provider]:
        allowed = ", ".join(sorted(ALLOWED_MODELS[provider]))
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model for provider {provider}. Allowed: {allowed}."
        )

    # --- Prepare arguments -------------------------------------------------
    llm_id = request.model_name
    user_prompt = "\n".join(request.messages)  # Merge list into a single prompt
    allow_search = request.allow_search
    system_prompt = request.system_prompt

    # --- Call the agent ----------------------------------------------------
    try:
        ai_response = get_response_from_ai_agent(
            llm_id=llm_id,
            query=user_prompt,
            allow_search=allow_search,
            system_prompt=system_prompt,
            provider=provider,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    return {"response": ai_response}

# 4. Run with: uvicorn api_server:app --host 127.0.0.1 --port 9999
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="127.0.0.1", port=9999, reload=True)
