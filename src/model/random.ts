/** Seeded innovation generator: xorshift32 + Box-Muller, version 1. */
export const RNG_ALGORITHM = "xorshift32-box-muller-v1";

export function seededNormals(seed: number, pathCount = 100, stepCount = 50): number[][] {
  if (!Number.isInteger(seed)) throw new TypeError("seed must be an integer");
  if (!Number.isInteger(pathCount) || pathCount <= 0) throw new RangeError("pathCount must be positive");
  if (!Number.isInteger(stepCount) || stepCount <= 0) throw new RangeError("stepCount must be positive");

  let state = seed >>> 0;
  if (state === 0) state = 0x6d2b79f5;
  let spare: number | undefined;
  const uniform = (): number => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return ((state >>> 0) + 0.5) / 0x1_0000_0000;
  };
  const normal = (): number => {
    if (spare !== undefined) {
      const value = spare;
      spare = undefined;
      return value;
    }
    const radius = Math.sqrt(-2 * Math.log(uniform()));
    const angle = 2 * Math.PI * uniform();
    spare = radius * Math.sin(angle);
    return radius * Math.cos(angle);
  };

  return Array.from({ length: pathCount }, () =>
    Array.from({ length: stepCount }, () => normal()),
  );
}
