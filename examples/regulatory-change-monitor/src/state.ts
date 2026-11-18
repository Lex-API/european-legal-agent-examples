import { existsSync, readFileSync, writeFileSync } from "node:fs";

/** What the monitor remembers between runs, so each change is reported once. */
export interface MonitorState {
  /** Item key -> when it was first seen (ISO). */
  seen: Record<string, string>;
  /** Watched acts whose existing relations were recorded as a baseline. */
  baselined: string[];
  /** Webhook delivery ids already handled (deliveries can be retried). */
  deliveries: string[];
}

export function emptyState(): MonitorState {
  return { seen: {}, baselined: [], deliveries: [] };
}

export function loadState(path: string): MonitorState {
  if (!existsSync(path)) return emptyState();
  return { ...emptyState(), ...(JSON.parse(readFileSync(path, "utf8")) as Partial<MonitorState>) };
}

export function saveState(path: string, state: MonitorState): void {
  writeFileSync(path, JSON.stringify(state, null, 2) + "\n");
}

/** Keep the items not reported before, and remember them. */
export function takeNew<T extends { key: string }>(state: MonitorState, items: T[], now: Date = new Date()): T[] {
  const fresh = items.filter((item) => !state.seen[item.key]);
  for (const item of fresh) state.seen[item.key] = now.toISOString();
  return fresh;
}
