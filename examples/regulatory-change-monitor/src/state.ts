import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { ConfigError } from "../../../shared/cli.ts";

/** What the monitor remembers between runs, so each change is reported once. */
export interface MonitorState {
  /** Item key -> when it was first seen (ISO). */
  seen: Record<string, string>;
  /** Watched acts whose existing relations were recorded as a baseline. */
  baselined: string[];
  /** Webhook delivery ids already handled (deliveries can be retried). Only the newest are kept. */
  deliveries: string[];
}

/** LexAPI retries a delivery for about a day, so a bounded memory of ids is enough. */
export const MAX_REMEMBERED_DELIVERIES = 10_000;

/** Record a delivery id; returns `false` when it was already handled. */
export function rememberDelivery(state: MonitorState, id: string): boolean {
  if (state.deliveries.includes(id)) return false;
  state.deliveries.push(id);
  if (state.deliveries.length > MAX_REMEMBERED_DELIVERIES) {
    state.deliveries.splice(0, state.deliveries.length - MAX_REMEMBERED_DELIVERIES);
  }
  return true;
}

export function emptyState(): MonitorState {
  return { seen: {}, baselined: [], deliveries: [] };
}

export function loadState(path: string): MonitorState {
  if (!existsSync(path)) return emptyState();
  const unreadable = new ConfigError(`The state file ${path} could not be read: it is not the JSON this monitor writes. Fix it, or remove it to start again from a baseline.`);
  let raw: unknown;
  try {
    raw = JSON.parse(readFileSync(path, "utf8"));
  } catch {
    throw unreadable;
  }
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw unreadable;
  const state = { ...emptyState(), ...(raw as Partial<MonitorState>) };
  const strings = (value: unknown) => Array.isArray(value) && value.every((item) => typeof item === "string");
  if (!state.seen || typeof state.seen !== "object" || Array.isArray(state.seen) || !strings(state.baselined) || !strings(state.deliveries)) {
    throw unreadable;
  }
  return state;
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
