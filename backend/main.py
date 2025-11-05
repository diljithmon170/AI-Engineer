from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI!"}

@app.get("/generate/{text}")
async def generate_text(text: str):
    return {"generated_text": f"You asked to generate: {text}"}
