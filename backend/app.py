from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from services.market import router as market_router
from services.stock_assessment import router as stock_router
from services.whale_watching import router as whale_router
from services.ml_training import router as ml_router
from services.ml_backtest import router as backtest_router
from services.csv_upload_router import router as csv_router
from services.categorizer_router import router as categorizer_router

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
app.include_router(ml_router, prefix="/api", tags=["ml"])
app.include_router(backtest_router, prefix="/api", tags=["backtest"])
app.include_router(csv_router, prefix="/api", tags=["csv"])
app.include_router(categorizer_router, prefix="/api", tags=["categorizer"])

@app.get("/")
def api_root():
    return {"message": "Welcome to the Finance Dashboard API"}