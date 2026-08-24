from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.config import CORS_ORIGINS, RESULTS_DIR
from api.routes.predict import router as predict_router

app = FastAPI(
    title="Brain Tumor Segmentation API",
    description="API for brain tumor segmentation using deep learning models",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/results",
    StaticFiles(directory=RESULTS_DIR),
    name="results",
)


@app.get("/")
def root():
    return {
        "message": "Brain Tumor Segmentation API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }



app.include_router(
    predict_router,
)
