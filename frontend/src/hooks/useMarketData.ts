import { useQuery } from "@tanstack/react-query";
import { fetchMarketData,  type MarketData } from "../services/api";

export function useMarketData(ticker: string) {
  return useQuery<MarketData>({
    queryKey: ["market", ticker],
    queryFn: () => fetchMarketData(ticker),
  });
}
