import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import DashboardLayout from "./layout/DashboardLayout";
import MarketOverview from "./pages/MarketOverview";
import Portfolio from "./pages/Portfolio";
import StockDetail from "./pages/StockDetail";
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';


const queryClient = new QueryClient();


function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <DashboardLayout>
          <Routes>
            <Route path="/" element={<MarketOverview />} />
            <Route path="/portfolio" element={<Portfolio />} />
            <Route path="/stock/:ticker" element={<StockDetail />} />
          </Routes>
        </DashboardLayout>
      </Router>
    </QueryClientProvider>
  );
}

export default App;
