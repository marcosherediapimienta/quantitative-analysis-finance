import React from "react";
import { PieChart, Pie, Cell, ResponsiveContainer } from "recharts";
import { motion } from "framer-motion";

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

const COLORS = ["#60a5fa", "#fbbf24", "#34d399"];

export default function Portfolio() {
  return (
    <div className="p-4 pb-20 max-w-xl mx-auto">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} className="mb-4">
        <div className="text-lg font-semibold mb-2">Tus posiciones</div>
        <div className="overflow-x-auto rounded-lg shadow">
          <table className="min-w-full text-sm bg-card dark:bg-zinc-800">
            <thead>
              <tr className="text-muted-foreground">
                <th className="px-3 py-2 text-left">Activo</th>
                <th className="px-3 py-2">% Portafolio</th>
                <th className="px-3 py-2">Rentab.</th>
                <th className="px-3 py-2">Riesgo</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((p) => (
                <tr key={p.asset} className="border-t border-border">
                  <td className="px-3 py-2 font-medium">{p.asset}</td>
                  <td className="px-3 py-2 text-center">{p.percent}%</td>
                  <td className="px-3 py-2 text-center">{p.return}</td>
                  <td className="px-3 py-2 text-center">{p.risk}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2, duration: 0.5 }} className="bg-card rounded-xl shadow p-4 mb-4 dark:bg-zinc-800">
        <div className="text-sm mb-2 font-medium">Distribución por categoría</div>
        <ResponsiveContainer width="100%" height={160}>
          <PieChart>
            <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={60} innerRadius={35} paddingAngle={4}>
              {pieData.map((entry, idx) => (
                <Cell key={`cell-${idx}`} fill={COLORS[idx % COLORS.length]} />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>
      </motion.div>
      <div className="flex gap-2 justify-center">
        <button className="bg-blue-500 text-white rounded-lg px-4 py-2 shadow hover:bg-blue-600 transition">Comprar</button>
        <button className="bg-green-500 text-white rounded-lg px-4 py-2 shadow hover:bg-green-600 transition">Vender</button>
        <button className="bg-yellow-500 text-white rounded-lg px-4 py-2 shadow hover:bg-yellow-600 transition">Rebalancear</button>
      </div>
    </div>
  );
}
