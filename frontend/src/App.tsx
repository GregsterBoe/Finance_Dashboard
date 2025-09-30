import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import DashboardLayout from "./layout/DashboardLayout";
import MarketOverview from "./pages/MarketOverview";
import Stocks from "./pages/Stocks";
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import WhaleWatchlist from "./pages/WhaleWatchlist";
import ActivityTracker from "./pages/ActivityTracker";


const queryClient = new QueryClient();


function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <DashboardLayout>
          <Routes>
            <Route path="/" element={<MarketOverview />} />
            <Route path="/stocks" element={<Stocks />} />
            <Route path="/whale-watchlist" element={<WhaleWatchlist />} />
            <Route path="/activity-tracker" element={<ActivityTracker />} />
          </Routes>
        </DashboardLayout>
      </Router>
    </QueryClientProvider>
  );
}

export default App;
