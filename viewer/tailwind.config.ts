import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#17202a",
        mist: "#eef2f6",
        line: "#d9e1ea",
        leaf: "#2f7d5a",
        coral: "#bf4f45",
        gold: "#a56a12",
        plum: "#6f4aa3",
      },
      boxShadow: {
        soft: "0 16px 40px rgba(23, 32, 42, 0.08)",
      },
    },
  },
  plugins: [],
};

export default config;
