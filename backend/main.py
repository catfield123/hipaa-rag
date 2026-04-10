from typing import Annotated

from fastapi import FastAPI, Path

app = FastAPI(title="HIPAA RAG Backend", root_path="/api")


@app.get("/hello/{name}")
def hello(name: Annotated[str, Path(min_length=1)]) -> dict[str, str]:
    return {"greeting": f"Hello, {name}!"}
