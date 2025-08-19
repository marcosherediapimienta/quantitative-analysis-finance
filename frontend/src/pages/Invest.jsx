import React from "react";
import { LineChart, Line, ResponsiveContainer } from "recharts";
import { motion } from "framer-motion";

const strategies = [
  {
    name: "Momentum",
    desc: "Invierte en activos con mejor desempeño reciente.",
    data: [
      { x: 0, y: 100 },
      { x: 1, y: 110 },
      { x: 2, y: 120 },
      { x: 3, y: 130 },
    ],
  },
  {
    name: "Value",
    desc: "Busca activos subvalorados con potencial de crecimiento.",
    data: [
      { x: 0, y: 100 },
      { x: 1, y: 105 },
      { x: 2, y: 112 },
      { x: 3, y: 120 },
    ],
  },
  {
    name: "Balanceado",
    desc: "Diversifica entre acciones, bonos y ETFs.",
    data: [
      { x: 0, y: 100 },
      { x: 1, y: 103 },
      { x: 2, y: 108 },
      { x: 3, y: 115 },
    ],
  },
];

export default function Invest() {
  return (
    <div className="p-4 pb-20 max-w-xl mx-auto">
      <div className="text-lg font-semibold mb-4">Estrategias de inversión</div>
      <div className="grid gap-4">
        {strategies.map((s, i) => (
          <motion.div key={s.name} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.1, duration: 0.5 }} className="bg-card rounded-xl shadow p-4 flex flex-col gap-2 dark:bg-zinc-800">
            <div className="flex items-center justify-between">
              <div className="font-bold text-base">{s.name}</div>
              <button className="bg-blue-500 text-white rounded-lg px-3 py-1 text-xs shadow hover:bg-blue-600 transition">Invertir en esta estrategia</button>
            </div>
            <div className="text-sm text-muted-foreground mb-2">{s.desc}</div>
            <ResponsiveContainer width="100%" height={60}>
              <LineChart data={s.data} margin={{ top: 5, right: 5, left: 0, bottom: 0 }}>
                <Line type="monotone" dataKey="y" stroke="#60a5fa" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
