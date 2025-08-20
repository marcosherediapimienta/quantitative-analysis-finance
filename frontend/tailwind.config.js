/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#0f172a",
        foreground: "#f1f5f9",
        card: "#18181b",
        border: "#27272a",
        primary: "#60a5fa",
        'muted-foreground': '#a1a1aa',
      },
    },
  },
  plugins: [],
};
