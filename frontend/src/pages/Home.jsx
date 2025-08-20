import React, { useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell,
} from "recharts";
import { motion, AnimatePresence } from "framer-motion";
import { Home, PieChart as PieIcon, TrendingUp, User, ShieldCheck, BarChart3, Layers3 } from "lucide-react";

// --- UI COMPONENTS (shadcn/ui style) ---
const Button = ({ children, className = "", ...props }) => (
  <button
    className={`px-4 py-2 rounded-2xl font-semibold bg-zinc-800 text-white shadow hover:bg-blue-600 transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-400/50 ${className}`}
    {...props}
  >
    {children}
  </button>
);

const Badge = ({ color = "bg-blue-500", children }) => (
  <span className={`inline-block px-2 py-0.5 rounded-full text-xs font-bold text-white ${color}`}>{children}</span>
);

const Card = ({ children, className = "" }) => (
  <div className={`bg-zinc-900/90 border border-zinc-800 rounded-2xl shadow-xl p-4 ${className}`}>{children}</div>
);

const TabIcon = ({ icon, active }) => (
  <span className={`flex items-center justify-center ${active ? "text-blue-400" : "text-zinc-400"}`}>{icon}</span>
);

// --- DATA ---
const portfolioLine = [
  { name: "Ene", portafolio: 10000, benchmark: 10000 },
  { name: "Feb", portafolio: 10200, benchmark: 10100 },
  { name: "Mar", portafolio: 10400, benchmark: 10250 },
  { name: "Abr", portafolio: 10700, benchmark: 10400 },
  { name: "May", portafolio: 11000, benchmark: 10600 },
  { name: "Jun", portafolio: 11250, benchmark: 10800 },
];

const positions = [
  { asset: "AAPL", percent: 35, return: "+14%", risk: "Medio" },
  { asset: "VOO", percent: 25, return: "+9%", risk: "Bajo" },
  { asset: "TSLA", percent: 15, return: "+22%", risk: "Alto" },
  { asset: "BND", percent: 25, return: "+3%", risk: "Bajo" },
];

const pieData = [
  { name: "Acciones", value: 50 },
  { name: "ETFs", value: 30 },
  { name: "Bonos", value: 20 },
];
const PIE_COLORS = ["#60a5fa", "#fbbf24", "#34d399"];

const strategies = [
  {
    name: "Momentum",
    desc: "Sigue tendencias del mercado",
    data: [
      { x: 0, y: 100 }, { x: 1, y: 110 }, { x: 2, y: 120 }, { x: 3, y: 130 },
    ],
    color: "#60a5fa",
  },
  {
    name: "Value",
    desc: "Invierte en empresas infravaloradas",
    data: [
      { x: 0, y: 100 }, { x: 1, y: 105 }, { x: 2, y: 112 }, { x: 3, y: 120 },
    ],
    color: "#fbbf24",
  },
  {
    name: "Balanceado",
    desc: "Diversificación moderada",
    data: [
      { x: 0, y: 100 }, { x: 1, y: 103 }, { x: 2, y: 108 }, { x: 3, y: 115 },
    ],
    color: "#34d399",
  },
];

// --- NAVIGATION ---
const NAV = [
  { label: "Inicio", icon: <Home size={22} />, key: "home" },
  { label: "Portafolio", icon: <PieIcon size={22} />, key: "portfolio" },
  { label: "Invertir", icon: <TrendingUp size={22} />, key: "invest" },
  { label: "Perfil", icon: <User size={22} />, key: "profile" },
];

// --- MAIN APP ---
export default function App() {
  const [tab, setTab] = useState("home");
  React.useEffect(() => {
    document.documentElement.classList.add("dark");
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-b from-zinc-950 via-zinc-900 to-zinc-950 text-white font-sans transition-colors duration-300">
      <main className="pb-24 max-w-md mx-auto">
        <AnimatePresence mode="wait">
          {tab === "home" && <HomeScreen key="home" />}
          {tab === "portfolio" && <PortfolioScreen key="portfolio" />}
          {tab === "invest" && <InvestScreen key="invest" />}
          {tab === "profile" && <ProfileScreen key="profile" />}
        </AnimatePresence>
      </main>
      <nav className="fixed bottom-0 left-0 right-0 z-50 bg-zinc-950/95 border-t border-zinc-800 flex justify-around items-center h-16 shadow-2xl backdrop-blur-md">
        {NAV.map((item) => (
          <button
            key={item.key}
            className="flex flex-col items-center text-xs font-medium focus:outline-none group"
            onClick={() => setTab(item.key)}
            aria-label={item.label}
          >
            <TabIcon icon={item.icon} active={tab === item.key} />
            <span className={`mt-1 ${tab === item.key ? "text-blue-400" : "text-zinc-400"} group-hover:text-blue-300 transition-colors`}>{item.label}</span>
          </button>
        ))}
      </nav>
    </div>
  );
}

// --- SCREENS ---
function HomeScreen() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 32 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 32 }}
      transition={{ duration: 0.5 }}
      className="px-4 pt-6 pb-4"
    >
      <Card className="mb-6 flex flex-col items-center gap-2 bg-gradient-to-br from-blue-900/60 to-zinc-900/80">
        <span className="uppercase tracking-widest text-xs text-blue-300 font-semibold">Saldo total invertido</span>
        <span className="text-4xl font-extrabold text-white drop-shadow-sm">$11,250</span>
        <span className="text-sm font-medium text-green-400 bg-green-900/30 px-2 py-0.5 rounded-full mt-1">Rentabilidad acumulada: +12.5%</span>
      </Card>
      <Card className="mb-6 p-4 bg-gradient-to-br from-zinc-900/80 to-zinc-800/80">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-semibold text-zinc-200">Evolución del portafolio</span>
          <span className="text-xs text-zinc-400">vs S&amp;P 500</span>
        </div>
        <ResponsiveContainer width="100%" height={160}>
          <LineChart data={portfolioLine}>
            <XAxis dataKey="name" hide />
            <YAxis hide />
            <Tooltip contentStyle={{ background: '#18181b', border: 'none', borderRadius: 8, color: '#fff' }} labelStyle={{ color: '#a1a1aa' }} />
            <Line type="monotone" dataKey="portafolio" stroke="#60a5fa" strokeWidth={3} dot={false} />
            <Line type="monotone" dataKey="benchmark" stroke="#fbbf24" strokeWidth={2} dot={false} strokeDasharray="4 4" />
          </LineChart>
        </ResponsiveContainer>
      </Card>
      <div className="grid grid-cols-3 gap-4">
        <Card className="flex flex-col items-center py-4 px-2 hover:scale-105 hover:shadow-lg transition-transform duration-200">
          <BarChart3 className="mb-1 text-blue-400" size={22} />
          <span className="text-xl font-bold text-blue-400">+12.5%</span>
          <span className="text-xs text-zinc-400 text-center font-medium tracking-wide">Rentabilidad YTD</span>
        </Card>
        <Card className="flex flex-col items-center py-4 px-2 hover:scale-105 hover:shadow-lg transition-transform duration-200">
          <ShieldCheck className="mb-1 text-yellow-400" size={22} />
          <span className="text-xl font-bold text-yellow-400">Medio</span>
          <span className="text-xs text-zinc-400 text-center font-medium tracking-wide">Riesgo</span>
        </Card>
        <Card className="flex flex-col items-center py-4 px-2 hover:scale-105 hover:shadow-lg transition-transform duration-200">
          <Layers3 className="mb-1 text-green-400" size={22} />
          <span className="text-xl font-bold text-green-400">Alta</span>
          <span className="text-xs text-zinc-400 text-center font-medium tracking-wide">Diversificación</span>
        </Card>
      </div>
    </motion.div>
  );
}

function PortfolioScreen() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 32 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 32 }}
      transition={{ duration: 0.5 }}
      className="px-4 pt-6 pb-4"
    >
      <Card className="mb-6 overflow-x-auto">
        <div className="text-lg font-semibold mb-3 text-zinc-100">Tus posiciones</div>
        <table className="min-w-full text-sm rounded-2xl overflow-hidden">
          <thead>
            <tr className="text-zinc-400 bg-zinc-900/80">
              <th className="px-3 py-2 text-left font-semibold">Activo</th>
              <th className="px-3 py-2 font-semibold">%</th>
              <th className="px-3 py-2 font-semibold">Rentab.</th>
              <th className="px-3 py-2 font-semibold">Riesgo</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((p) => (
              <tr key={p.asset} className="border-t border-zinc-800 hover:bg-zinc-800/40 transition">
                <td className="px-3 py-2 font-medium text-zinc-200">{p.asset}</td>
                <td className="px-3 py-2 text-center">{p.percent}%</td>
                <td className="px-3 py-2 text-center">{p.return}</td>
                <td className="px-3 py-2 text-center">
                  <Badge color={
                    p.risk === "Alto" ? "bg-red-500" :
                    p.risk === "Medio" ? "bg-yellow-500" : "bg-green-500"
                  }>
                    {p.risk}
                  </Badge>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
      <Card className="mb-6 flex flex-col items-center">
        <div className="text-sm mb-2 font-medium text-zinc-200">Distribución por categoría</div>
        <ResponsiveContainer width="100%" height={160}>
          <PieChart>
            <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={60} innerRadius={35} paddingAngle={4}>
              {pieData.map((entry, idx) => (
                <Cell key={`cell-${idx}`} fill={PIE_COLORS[idx % PIE_COLORS.length]} />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>
        <div className="flex gap-3 mt-2">
          {pieData.map((d, i) => (
            <span key={d.name} className="flex items-center gap-1 text-xs text-zinc-300">
              <span className="inline-block w-3 h-3 rounded-full" style={{ background: PIE_COLORS[i] }}></span>
              {d.name}
            </span>
          ))}
        </div>
      </Card>
      <div className="flex gap-2 justify-center">
        <Button className="bg-blue-500 hover:bg-blue-600">Comprar</Button>
        <Button className="bg-green-500 hover:bg-green-600">Vender</Button>
        <Button className="bg-yellow-500 hover:bg-yellow-600 text-zinc-900">Rebalancear</Button>
      </div>
    </motion.div>
  );
}

function InvestScreen() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 32 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 32 }}
      transition={{ duration: 0.5 }}
      className="px-4 pt-6 pb-4"
    >
      <div className="text-lg font-semibold mb-4 text-zinc-100">Estrategias de inversión</div>
      <div className="grid gap-4">
        {strategies.map((s, i) => (
          <motion.div
            key={s.name}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1, duration: 0.5 }}
          >
            <Card className="flex flex-col gap-2">
              <div className="flex items-center justify-between">
                <div className="font-bold text-base text-zinc-200">{s.name}</div>
                <Button className="bg-blue-500 hover:bg-blue-600 text-xs px-3 py-1 rounded-xl">Invertir en esta estrategia</Button>
              </div>
              <div className="text-sm text-zinc-400 mb-2">{s.desc}</div>
              <ResponsiveContainer width="100%" height={50}>
                <LineChart data={s.data}>
                  <Line type="monotone" dataKey="y" stroke={s.color} strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}

function ProfileScreen() {
  const [risk, setRisk] = useState(2);
  const [autoInvest, setAutoInvest] = useState(true);

  return (
    <motion.div
      initial={{ opacity: 0, y: 32 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 32 }}
      transition={{ duration: 0.5 }}
      className="px-4 pt-6 pb-4"
    >
      <Card className="mb-6">
        <div className="text-lg font-semibold mb-2 text-zinc-100">Preferencias de riesgo</div>
        <input
          type="range"
          min={1}
          max={3}
          step={1}
          value={risk}
          onChange={e => setRisk(Number(e.target.value))}
          className="w-full accent-blue-500 h-2 rounded-lg appearance-none bg-zinc-800"
        />
        <div className="flex justify-between text-xs mt-1 text-zinc-400">
          <span className={risk === 1 ? "text-blue-400 font-bold" : ""}>Bajo</span>
          <span className={risk === 2 ? "text-yellow-400 font-bold" : ""}>Medio</span>
          <span className={risk === 3 ? "text-red-400 font-bold" : ""}>Alto</span>
        </div>
      </Card>
      <Card className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <div className="font-medium text-zinc-200">Aportaciones automáticas</div>
          <label className="inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={autoInvest}
              onChange={e => setAutoInvest(e.target.checked)}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-zinc-700 rounded-full peer peer-checked:bg-blue-500 transition-all duration-200 relative">
              <div className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow transition-all duration-200 ${autoInvest ? 'translate-x-5 bg-blue-400' : ''}`}></div>
            </div>
          </label>
        </div>
        <div className="text-xs text-zinc-400">Activa o desactiva las aportaciones periódicas a tu portafolio.</div>
      </Card>
      <Card>
        <div className="text-lg font-semibold mb-2 text-zinc-100">Historial de movimientos</div>
        <ul className="space-y-2 text-sm">
          <li>+ $500 <span className="text-green-400">Aporte automático</span> <span className="text-zinc-400">(01/08/2025)</span></li>
          <li>- $200 <span className="text-red-400">Retiro</span> <span className="text-zinc-400">(15/07/2025)</span></li>
          <li>+ $300 <span className="text-green-400">Aporte manual</span> <span className="text-zinc-400">(01/07/2025)</span></li>
        </ul>
      </Card>
    </motion.div>
  );
}
