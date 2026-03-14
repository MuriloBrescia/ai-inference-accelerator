from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from inference_engine import InferenceEngine
import asyncio

app = FastAPI(title="AI Inference Accelerator")
engine = InferenceEngine(model_name="distilgpt2")

class Query(BaseModel):
    prompt: str
    max_length: int = 50

@app.post("/generate")
async def generate(query: Query):
    try:
        result = await engine.infer(query.prompt, query.max_length)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream")
async def stream(query: Query):
    return StreamingResponse(engine.stream_infer(query.prompt, query.max_length), media_type="text/plain")