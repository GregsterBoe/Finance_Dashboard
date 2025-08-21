import type { ReactNode } from "react";
import { Link } from "react-router-dom";

interface Props {
  children: ReactNode;
}

export default function DashboardLayout({ children }: Props) {
  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <aside className="w-64 bg-gray-900 text-white p-4">
        <h2 className="text-xl font-bold mb-6">Finance Dashboard</h2>
        <nav className="flex flex-col gap-4">
          <Link to="/">📊 Market Overview</Link>
          <Link to="/portfolio">💼 Portfolio</Link>
        </nav>
      </aside>

      {/* Content */}
      <main className="flex-1 p-6 bg-gray-50 overflow-y-auto">{children}</main>
    </div>
  );
}
