import axios from "axios";

const API_BASE = "http://localhost:8000"; // Your FastAPI backend

export interface MarketData {
  ticker: string;
  prices: { date: string; value: number }[];
}

export async function fetchMarketData(ticker: string): Promise<MarketData> {
  const res = await axios.get(`${API_BASE}/market/${ticker}`);
  return res.data;
}
