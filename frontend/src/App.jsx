import React, { useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import {
  Home,
  PieChart as PieIcon,
  TrendingUp,
  User,
  BarChart3,
  ShieldCheck,
  Layers3,
  LineChart as LineChartIcon,
  CandlestickChart,
  FileBarChart,
  Sigma
} from "lucide-react";

// shadcn-like primitives (inline)
const cx = (...classes) => classes.filter(Boolean).join(" ");

const Card = ({ className = "", children }) => (
  <div className={cx(
    "rounded-2xl border border-zinc-800 bg-zinc-900/80 shadow-2xl backdrop-blur-sm",
    "text-zinc-100",
    className
  )}>{children}</div>
);

const CardContent = ({ className = "", children }) => (
  <div className={cx("p-4", className)}>{children}</div>
);

const CardHeader = ({ className = "", children }) => (
  <div className={cx("p-4 pb-0", className)}>{children}</div>
);

const CardTitle = ({ className = "", children }) => (
  <h3 className={cx("text-base font-semibold text-zinc-200", className)}>{children}</h3>
);

const Button = ({ className = "", variant = "default", children, ...props }) => {
  const variants = {
    default: "bg-blue-500 hover:bg-blue-600 text-white",
    ghost: "bg-transparent hover:bg-zinc-800/60 text-zinc-200",
  };
  return (
    <button
      className={cx(
        "inline-flex items-center justify-center whitespace-nowrap",
        "rounded-2xl px-4 py-2 text-sm font-semibold",
        "transition-colors focus:outline-none focus:ring-2 focus:ring-blue-400/40",
        variants[variant],
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
};

const Input = ({ className = "", ...props }) => (
  <input
    className={cx(
      "w-full rounded-xl border border-zinc-800 bg-zinc-900/60 px-3 py-2 text-sm text-zinc-100",
      "placeholder:text-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-400/40",
      className
    )}
    {...props}
  />
);

const Label = ({ children, className = "" }) => (
  <label className={cx("text-xs font-semibold text-zinc-400", className)}>{children}</label>
);

const Select = ({ className = "", children, ...props }) => (
  <select
    className={cx(
      "w-full rounded-xl border border-zinc-800 bg-zinc-900/60 px-3 py-2 text-sm text-zinc-100",
      "focus:outline-none focus:ring-2 focus:ring-blue-400/40",
      className
    )}
    {...props}
  >
    {children}
  </select>
);

const Badge = ({ className = "", color = "blue", children }) => {
  const colorMap = {
    blue: "bg-blue-500 text-white",
    yellow: "bg-yellow-500 text-zinc-900",
    green: "bg-green-500 text-zinc-900",
    red: "bg-red-500 text-white",
  };
  return (
    <span className={cx(
      "inline-flex items-center rounded-full px-2 py-0.5 text-xs font-bold",
      colorMap[color] || colorMap.blue,
      className
    )}>
      {children}
    </span>
  );
};

const BottomNav = ({ current, onChange }) => {
  const items = [
    { key: "home", label: "Inicio", icon: <Home size={22} /> },
    { key: "opciones", label: "Opciones", icon: <Sigma size={22} /> },
    { key: "fundamental", label: "Fundamental", icon: <FileBarChart size={22} /> },
    { key: "tecnico", label: "Técnico", icon: <CandlestickChart size={22} /> },
    { key: "profile", label: "Perfil", icon: <User size={22} /> },
  ];

  return (
    <nav
      className={cx(
        "fixed inset-x-0 bottom-0 z-50 md:hidden",
        "border-t border-zinc-800 bg-zinc-950/90 backdrop-blur-md",
        "shadow-[0_-8px_24px_rgba(0,0,0,0.4)]"
      )}
      style={{ paddingBottom: "env(safe-area-inset-bottom)" }}
    >
      <div className="mx-auto grid max-w-md grid-cols-5">
        {items.map((item) => {
          const active = current === item.key;
          return (
            <button
              key={item.key}
              onClick={() => onChange(item.key)}
              className="flex flex-col items-center gap-1 py-2"
              aria-label={item.label}
            >
              <span className={cx("transition-colors", active ? "text-blue-400" : "text-zinc-400")}>{item.icon}</span>
              <span
                className={cx(
                  "text-[11px] font-medium transition-colors",
                  active ? "text-blue-400" : "text-zinc-400"
                )}
              >
                {item.label}
              </span>
            </button>
          );
        })}
      </div>
    </nav>
  );
};

// Data
const usePortfolioData = () => {
  return useMemo(
    () => [
      { name: "Ene", portfolio: 10000, sp500: 10000 },
      { name: "Feb", portfolio: 10200, sp500: 10100 },
      { name: "Mar", portfolio: 10400, sp500: 10250 },
      { name: "Abr", portfolio: 10700, sp500: 10400 },
      { name: "May", portfolio: 11000, sp500: 10600 },
      { name: "Jun", portfolio: 11250, sp500: 10800 },
    ],
    []
  );
};

export default function App() {
  const [tab, setTab] = useState("home");
  useEffect(() => {
    document.documentElement.classList.add("dark");
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-b from-zinc-950 via-zinc-900 to-zinc-950 text-white">
      <header className="sticky top-0 z-40 border-b border-zinc-800 bg-zinc-950/80 backdrop-blur-md">
        <div className="mx-auto max-w-md md:max-w-2xl lg:max-w-5xl px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="h-6 w-6 rounded-md bg-blue-500/20 ring-1 ring-blue-400/40 flex items-center justify-center">
              <span className="text-blue-300 text-xs font-black">GA</span>
            </div>
            <span className="text-sm font-semibold tracking-wide text-zinc-100">GalaAnalytics</span>
          </div>
          <div className="hidden md:flex items-center gap-2">
            <Button variant="ghost" onClick={() => setTab("opciones")}>Opciones</Button>
            <Button variant="ghost" onClick={() => setTab("fundamental")}>Fundamental</Button>
            <Button variant="ghost" onClick={() => setTab("tecnico")}>Técnico</Button>
            <Button variant="ghost" onClick={() => setTab("profile")}>Perfil</Button>
            <Badge color="blue">beta</Badge>
          </div>
          <Badge color="blue" className="md:hidden">beta</Badge>
        </div>
      </header>
      <main className="mx-auto max-w-md md:max-w-2xl lg:max-w-5xl pb-24 px-0 md:px-4">
        <AnimatePresence mode="wait">
          {tab === "home" && <HomeScreen key="home" onNavigate={setTab} />}
          {tab === "opciones" && <OptionsAnalysis key="opciones" />}
          {tab === "fundamental" && <ComingSoon key="fundamental" tab={"fundamental"} />} 
          {tab === "tecnico" && <ComingSoon key="tecnico" tab={"tecnico"} />} 
          {tab === "profile" && <ComingSoon key="profile" tab={"profile"} />}
        </AnimatePresence>
      </main>
      <BottomNav current={tab} onChange={setTab} />
    </div>
  );
}

function HomeScreen({ onNavigate }) {
  const chartData = usePortfolioData();

  return (
    <motion.div
      className="px-4 pt-6 pb-4"
      initial={{ opacity: 0, y: 32 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 32 }}
      transition={{ duration: 0.45 }}
    >
      <motion.div
        initial={{ opacity: 0, scale: 0.98 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.45 }}
      >
        <Card className="mb-6 bg-gradient-to-br from-blue-900/40 to-zinc-900/70">
          <CardContent className="flex flex-col items-center md:items-start gap-2 md:gap-3">
            <span className="text-xs font-semibold uppercase tracking-widest text-blue-300">
              Bienvenido a GalaAnalytics
            </span>
            <span className="text-3xl md:text-4xl font-extrabold text-white drop-shadow-sm text-center md:text-left">
              Opciones, fundamental y técnico en un solo lugar
            </span>
            <Badge color="green" className="mt-1">Tu centro de análisis financiero</Badge>
          </CardContent>
        </Card>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45, delay: 0.05 }}
      >
        <Card className="mb-6">
          <CardHeader className="flex items-center justify-between">
            <CardTitle>Visión rápida</CardTitle>
            <span className="text-xs text-zinc-400">Portafolio vs S&P 500</span>
          </CardHeader>
          <CardContent className="pt-2">
            <div className="h-[180px] md:h-[240px] lg:h-[280px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 6, right: 8, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="lineBlue" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#60a5fa" stopOpacity={0.9} />
                      <stop offset="100%" stopColor="#60a5fa" stopOpacity={0.2} />
                    </linearGradient>
                    <linearGradient id="lineAmber" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#fbbf24" stopOpacity={0.9} />
                      <stop offset="100%" stopColor="#fbbf24" stopOpacity={0.2} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="name" tick={{ fill: "#a1a1aa", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis hide />
                  <Tooltip
                    contentStyle={{ background: "#0a0a0a", border: "1px solid #27272a", borderRadius: 12 }}
                    labelStyle={{ color: "#a1a1aa" }}
                    itemStyle={{ color: "#e5e7eb" }}
                  />
                  <Legend wrapperStyle={{ paddingTop: 6 }} iconType="circle" iconSize={8} formatter={(value) => (
                    <span style={{ color: "#d4d4d8", fontSize: 12 }}>{value === "portfolio" ? "Portafolio" : "S&P 500"}</span>
                  )} />
                  <Line type="monotone" dataKey="portfolio" stroke="#60a5fa" strokeWidth={3} dot={false} />
                  <Line type="monotone" dataKey="sp500" stroke="#fbbf24" strokeWidth={2} dot={false} strokeDasharray="6 4" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 lg:gap-6">
        {[
          {
            key: "opciones",
            title: "Análisis de Opciones",
            subtitle: "Volatilidad, griegas y pricing",
            icon: <Sigma size={24} className="text-blue-400" />,
          },
          {
            key: "fundamental",
            title: "Análisis Fundamental",
            subtitle: "Ratios, flujos y valoración",
            icon: <FileBarChart size={24} className="text-amber-400" />,
          },
          {
            key: "tecnico",
            title: "Análisis Técnico",
            subtitle: "Tendencias y patrones",
            icon: <CandlestickChart size={24} className="text-emerald-400" />,
          },
          {
            key: "profile",
            title: "Perfil",
            subtitle: "Preferencias y cuenta",
            icon: <User size={24} className="text-zinc-300" />,
          },
        ].map((section, index) => (
          <motion.div
            key={section.key}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.05 * (index + 1) }}
          >
            <button
              onClick={() => onNavigate(section.key)}
              className="w-full text-left"
            >
              <Card className="hover:shadow-xl hover:-translate-y-0.5 transition-all duration-200">
                <CardContent className="flex items-center gap-3 py-4 px-3">
                  {section.icon}
                  <div className="flex flex-col">
                    <span className="text-sm font-semibold text-zinc-100">{section.title}</span>
                    <span className="text-xs text-zinc-400">{section.subtitle}</span>
                  </div>
                </CardContent>
              </Card>
            </button>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}

function ComingSoon({ tab }) {
  const titles = {
    opciones: "Análisis de Opciones",
    fundamental: "Análisis Fundamental",
    tecnico: "Análisis Técnico",
    profile: "Perfil",
  };
  return (
    <motion.div
      className="px-4 pt-6 pb-4"
      initial={{ opacity: 0, y: 32 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 32 }}
      transition={{ duration: 0.45 }}
    >
      <Card className="bg-gradient-to-br from-zinc-900/80 to-zinc-800/80">
        <CardContent className="flex flex-col items-center gap-3 py-10">
          <div className="rounded-2xl bg-zinc-800/80 px-3 py-1 text-xs font-semibold text-zinc-300">
            {titles[tab]}
          </div>
          <div className="text-2xl font-bold text-zinc-100">Próximamente</div>
          <div className="max-w-[26ch] text-center text-sm text-zinc-400">
            Estamos trabajando en esta sección para darte una experiencia de análisis aún mejor.
          </div>
          <Button variant="ghost" className="mt-2" onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}>
            Volver arriba
          </Button>
        </CardContent>
      </Card>
    </motion.div>
  );
}

function OptionsAnalysis() {
  const [symbol, setSymbol] = useState("AAPL");
  const [expirations, setExpirations] = useState([]);
  const [expFrom, setExpFrom] = useState("");
  const [expTo, setExpTo] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [chains, setChains] = useState({});

  useEffect(() => {
    fetchExpirations(symbol);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const apiBase = (() => {
    const env = typeof window !== "undefined" ? window.location : { hostname: "localhost" };
    // Default local backend
    return (import.meta && import.meta.env && import.meta.env.VITE_API_BASE) || `http://localhost:8000`;
  })();

  async function fetchExpirations(sym) {
    try {
      setError("");
      const res = await fetch(`${apiBase}/yahoo/options/expirations?symbol=${encodeURIComponent(sym)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setExpirations(data.expirations || []);
    } catch (e) {
      setError(`Error cargando expiraciones: ${e}`);
    }
  }

  async function fetchChains() {
    try {
      setLoading(true);
      setError("");
      const qs = new URLSearchParams({ symbol });
      if (expFrom) qs.append("from", expFrom);
      if (expTo) qs.append("to", expTo);
      const res = await fetch(`${apiBase}/yahoo/options/chains?${qs.toString()}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setChains(data.chains || {});
    } catch (e) {
      setError(`Error cargando cadenas: ${e}`);
    } finally {
      setLoading(false);
    }
  }

  const totalContracts = Object.values(chains).reduce((acc, c) => acc + (c.calls?.length || 0) + (c.puts?.length || 0), 0);

  return (
    <motion.div
      className="px-4 pt-6 pb-4"
      initial={{ opacity: 0, y: 32 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 32 }}
      transition={{ duration: 0.45 }}
    >
      <Card className="mb-4">
        <CardContent className="grid grid-cols-1 md:grid-cols-4 gap-3">
          <div>
            <Label>Ticker (Yahoo)</Label>
            <Input
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              placeholder="AAPL"
            />
          </div>
          <div>
            <Label>Desde (expiración)</Label>
            <Input type="date" value={expFrom} onChange={(e) => setExpFrom(e.target.value)} />
          </div>
          <div>
            <Label>Hasta (expiración)</Label>
            <Input type="date" value={expTo} onChange={(e) => setExpTo(e.target.value)} />
          </div>
          <div className="flex items-end gap-2">
            <Button onClick={() => { fetchExpirations(symbol); }}>Cargar expiraciones</Button>
            <Button onClick={fetchChains} className="min-w-36">Buscar</Button>
          </div>
        </CardContent>
      </Card>

      {error && (
        <Card className="mb-4 border-red-900/60">
          <CardContent className="text-red-300 text-sm">{error}</CardContent>
        </Card>
      )}

      {expirations.length > 0 && (
        <Card className="mb-4">
          <CardHeader>
            <CardTitle>Expiraciones disponibles</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {expirations.map((d) => (
                <span key={d} className="rounded-xl bg-zinc-800/60 px-2 py-1 text-xs text-zinc-300">
                  {d}
                </span>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {loading && (
        <Card className="mb-4">
          <CardContent className="text-zinc-300 text-sm">Cargando cadenas de opciones…</CardContent>
        </Card>
      )}

      {!loading && totalContracts > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Resultados ({totalContracts} contratos)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(chains).map(([exp, ch]) => (
                <div key={exp} className="rounded-xl border border-zinc-800">
                  <div className="flex items-center justify-between px-3 py-2 border-b border-zinc-800">
                    <span className="text-sm font-semibold text-zinc-200">{exp}</span>
                    <Badge color="blue">{(ch.calls?.length || 0) + (ch.puts?.length || 0)} contratos</Badge>
                  </div>
                  <div className="max-h-72 overflow-auto divide-y divide-zinc-800">
                    {ch.calls?.slice(0, 50).map((c, i) => (
                      <div key={`c-${i}`} className="flex items-center justify-between px-3 py-2 text-xs text-zinc-300">
                        <span>Call {c.contractSymbol}</span>
                        <span>Strike {c.strike}</span>
                        <span>IV {c.impliedVolatility}</span>
                      </div>
                    ))}
                    {ch.puts?.slice(0, 50).map((p, i) => (
                      <div key={`p-${i}`} className="flex items-center justify-between px-3 py-2 text-xs text-zinc-300">
                        <span>Put {p.contractSymbol}</span>
                        <span>Strike {p.strike}</span>
                        <span>IV {p.impliedVolatility}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </motion.div>
  );
}