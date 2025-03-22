import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  base: "https://shivam-daksh.github.io/full-stack-ml-image-classification/",
  optimizeDeps: {
    exclude: ["lucide-react"],
  },
});
