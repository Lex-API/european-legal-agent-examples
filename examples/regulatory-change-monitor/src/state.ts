import { chmodSync, lstatSync, readFileSync, readdirSync, readlinkSync, realpathSync, renameSync, rmSync, statSync, writeFileSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
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

function errorCode(error: unknown): string {
  const code = (error as { code?: unknown }).code;
  return typeof code === "string" ? code : "error";
}

/** The file the state is written to: a symlink's target (so the link stays a link), else the path itself. */
export function stateTarget(path: string): string {
  try {
    return realpathSync(path);
  } catch {
    try {
      // A link whose target does not exist yet: write that target.
      if (lstatSync(path).isSymbolicLink()) return resolve(dirname(path), readlinkSync(path));
    } catch {
      // not there: checked by loadState
    }
    return path;
  }
}

/** This monitor's temporary file for a state file: `.<name>.<pid>.tmp` next to it. */
const TEMPORARY = (target: string) => new RegExp(`^\\.${basename(target).replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\.\\d+\\.tmp$`);

/**
 * Read the state file before any request. A path that does not exist gives an empty state; one
 * that cannot be checked (a directory that cannot be searched), is a directory, cannot be read
 * or is not the JSON this monitor writes is a configuration error. Leftover temporary files of a
 * killed run for this state file are removed.
 */
export function loadState(path: string): MonitorState {
  const target = stateTarget(path);
  let exists = true;
  try {
    if (statSync(target).isDirectory()) throw new ConfigError(`The state file ${path} is a directory; give --state a file path.`);
  } catch (error) {
    if (error instanceof ConfigError) throw error;
    // ENOTDIR: a part of the path is a file; the directory check below says which.
    if (errorCode(error) !== "ENOENT" && errorCode(error) !== "ENOTDIR") {
      throw new ConfigError(`The state file ${path} cannot be checked (${errorCode(error)}).`);
    }
    exists = false;
  }
  try {
    const folder = statSync(dirname(target));
    if (!folder.isDirectory()) throw new ConfigError(`The state file ${path} cannot be written: ${dirname(target)} is not a directory.`);
  } catch (error) {
    if (error instanceof ConfigError) throw error;
    throw new ConfigError(`The state file ${path} cannot be checked: its directory ${dirname(target)} ${errorCode(error) === "ENOENT" ? "does not exist" : `cannot be read (${errorCode(error)})`}.`);
  }
  removeStaleTemporaries(target);
  if (!exists) return emptyState();
  let text: string;
  try {
    text = readFileSync(target, "utf8");
  } catch (error) {
    throw new ConfigError(`The state file ${path} could not be read (${errorCode(error)}).`);
  }
  const unreadable = new ConfigError(`The state file ${path} is not the JSON this monitor writes. Fix it, or remove it to start again from a baseline.`);
  let raw: unknown;
  try {
    raw = JSON.parse(text);
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

/** Remove this monitor's leftover temporary files for this state file (a killed run's); nothing else. */
function removeStaleTemporaries(target: string): void {
  const pattern = TEMPORARY(target);
  let names: string[];
  try {
    names = readdirSync(dirname(target));
  } catch {
    return;
  }
  for (const name of names) {
    if (pattern.test(name) && name !== basename(temporaryFor(target))) rmSync(join(dirname(target), name), { force: true });
  }
}

function temporaryFor(target: string): string {
  return join(dirname(target), `.${basename(target)}.${process.pid}.tmp`);
}

/**
 * Write the state file atomically: a temporary file next to the target, then a rename, so the
 * file is never half-written. A symlinked path keeps its link (the target is written). An
 * existing file keeps its mode; a new one is created readable by its owner only (0600), since
 * the state shows what is being watched. The monitor is meant to run one at a time.
 */
export function saveState(path: string, state: MonitorState): void {
  const target = stateTarget(path);
  const temporary = temporaryFor(target);
  let mode = 0o600;
  try {
    mode = statSync(target).mode & 0o777;
  } catch {
    // a new file
  }
  try {
    writeFileSync(temporary, JSON.stringify(state, null, 2) + "\n", { mode });
    chmodSync(temporary, mode); // the umask may have narrowed it
    renameSync(temporary, target);
  } catch (error) {
    rmSync(temporary, { force: true });
    throw error;
  }
}

/** The line for a state file that could not be written; the digest was printed before it. */
export function stateNotSaved(path: string, error: unknown): string {
  return (
    `The state file ${path} could not be written (${errorCode(error)}). ` +
    "The digest above was produced, but the state was not saved, so the next run will report the same items again."
  );
}

/** Keep the items not reported before, and remember them. */
export function takeNew<T extends { key: string }>(state: MonitorState, items: T[], now: Date = new Date()): T[] {
  const fresh = items.filter((item) => !state.seen[item.key]);
  for (const item of fresh) state.seen[item.key] = now.toISOString();
  return fresh;
}
