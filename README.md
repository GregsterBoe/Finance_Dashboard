# Finance Dashboard Frontend

This is the frontend for the Finance Dashboard project, built with React, TypeScript, and Tailwind CSS. It provides interactive dashboards and tools for stock market analysis, whale activity tracking, and machine learning-based price prediction.

## Project Structure

- `src/`
  - `pages/`  
    Contains main dashboard pages:
    - `MarketOverview.tsx`: Global market summary and regional breakdowns.
    - `Stocks.tsx`: Stock assessment panels and charts.
    - `WhaleWatchlist.tsx`: Institutional activity and volume ratio watchlist.
    - `ActivityTracker.tsx`: Symbol-level whale activity and dark pool signal analysis.
    - `StockPricePredictor.tsx`: ML model training and prediction UI.
    - `ModelBackTesting.tsx`: Model backtesting interface.
  - `hooks/`  
    Custom React hooks for data fetching.
  - `services/`  
    API service functions for backend communication.
  - `layout/`  
    Shared dashboard layout and navigation.
  - `main.tsx`, `App.tsx`  
    App entry point and routing.

- `public/`  
  Static assets.

- `index.html`  
  Main HTML file.

- `tailwind.config.js`, `postcss.config.cjs`  
  Styling configuration.

## Basic Functionalities

- **Market Overview:**  
  View global market indices, trends, and sentiment.

- **Stock Assessment:**  
  Analyze stocks by quality tiers, fundamentals, and price history.

- **Whale Watchlist:**  
  Track stocks with high institutional interest and unusual volume.

- **Activity Tracking:**  
  Monitor real-time whale activity and dark pool signals for specific stocks.

- **Price Prediction:**  
  Predict stock prices using machine learning models and historical data.

- **Model Backtesting:**  
  Test and validate machine learning models against historical data.

## Getting Started

To run the project locally:

1. Clone the repository.
2. Install dependencies: in frontend: `npm install` in backend: `uv pip install -r requirements.txt`.
3. Start the development server in frontend: `npm run dev` (for debugging backend: `npm run dev:debug`).
4. Open `http://localhost:5173` in your browser.

## Technologies Used

- **Frontend:**  
  - React
  - TypeScript
  - Tailwind CSS
  - Vite

- **Backend:**
  - uvicorn
  - FastAPI
  - yfinance

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

