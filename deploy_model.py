from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn
from typing import Optional

app = FastAPI(title="Healthcare LLM API")

class Query(BaseModel):
    question: str
    context: Optional[str] = None

class Response(BaseModel):
    answer: str
    confidence: float

# Load the trained model
model_path = "healthcare_llm"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_response(question: str, context: Optional[str] = None) -> tuple[str, float]:
    """Generate response from the healthcare LLM"""
    prompt = f"""You are a medical expert. Please provide a detailed and accurate response to the following medical question:

Question: {question}

Context: {context if context else ''}

Answer:"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=512,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Calculate confidence (placeholder - implement actual confidence calculation)
    confidence = 0.85
    
    return response, confidence

@app.post("/query", response_model=Response)
async def process_query(query: Query):
    try:
        answer, confidence = generate_response(query.question, query.context)
        return Response(answer=answer, confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 