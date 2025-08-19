import React from "react";

const Navbar = ({ current, setCurrent, navItems }) => (
  <nav className="fixed bottom-0 left-0 right-0 z-50 bg-background border-t border-border flex justify-around items-center h-16 shadow-lg dark:bg-zinc-900 dark:border-zinc-800">
    {navItems.map((item, idx) => (
      <button
        key={item.key}
        className={`flex flex-col items-center text-xs font-medium transition-colors duration-200 focus:outline-none ${
          current === item.key
            ? "text-primary dark:text-blue-400"
            : "text-muted-foreground dark:text-zinc-400"
        }`}
        onClick={() => setCurrent(item.key)}
        aria-label={item.label}
      >
        {item.icon}
        <span className="mt-1">{item.label}</span>
      </button>
    ))}
  </nav>
);

export default Navbar;
