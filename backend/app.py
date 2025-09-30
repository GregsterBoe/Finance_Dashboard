from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from services.market import router as market_router
from services.stock_assesment import router as stock_router
from services.whale_watching import router as whale_router

app = FastAPI()

# More permissive CORS settings for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],  # Expose all headers
)

app.include_router(market_router, prefix="/api")
app.include_router(stock_router, prefix="/api")
app.include_router(whale_router, prefix="/api")

@app.get("/")
def api_root():
    return {"message": "Welcome to the Finance Dashboard API"}