import { defineConfig } from "vite";

export default defineConfig({
  // Relative asset URLs work both at the domain root and at a Pages subpath.
  base: "./",
  worker: { format: "es" },
});
