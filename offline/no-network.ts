/**
 * Preloaded (`node --import`) into every example process the tests start:
 * any request that is not to 127.0.0.1 fails, so neither SDK (LexAPI or the
 * model provider) can reach a real service from the child process either.
 */
import { blockExternalFetch } from "./guard.ts";

blockExternalFetch();
