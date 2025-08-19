import React, { useState } from "react";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { motion } from "framer-motion";

export default function Profile() {
  const [risk, setRisk] = useState(2);
  const [autoInvest, setAutoInvest] = useState(true);

  return (
    <div className="p-4 pb-20 max-w-xl mx-auto">
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} className="mb-6">
        <div className="text-lg font-semibold mb-2">Preferencias de riesgo</div>
        <Slider min={1} max={3} step={1} value={[risk]} onValueChange={([v]) => setRisk(v)} className="w-full" />
        <div className="flex justify-between text-xs mt-1 text-muted-foreground">
          <span>Bajo</span>
          <span>Medio</span>
          <span>Alto</span>
        </div>
      </motion.div>
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2, duration: 0.5 }} className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <div className="font-medium">Aportaciones automáticas</div>
          <Switch checked={autoInvest} onCheckedChange={setAutoInvest} />
        </div>
        <div className="text-xs text-muted-foreground">Activa o desactiva las aportaciones periódicas a tu portafolio.</div>
      </motion.div>
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4, duration: 0.5 }}>
        <div className="text-lg font-semibold mb-2">Historial de movimientos</div>
        <ul className="bg-card rounded-xl shadow p-4 space-y-2 text-sm dark:bg-zinc-800">
          <li>+ $500 - Aporte automático (01/08/2025)</li>
          <li>- $200 - Retiro (15/07/2025)</li>
          <li>+ $300 - Aporte manual (01/07/2025)</li>
        </ul>
      </motion.div>
    </div>
  );
}
