/**
 * Shared setup for the TypeScript tests: start the stub, point the
 * environment at it, and refuse to continue if anything could reach a real
 * service.
 */
import { assertLocalBaseUrl, blockExternalFetch } from "./guard.ts";
import { startStubServer, type StubOptions, type StubServer } from "./stub-server.ts";

blockExternalFetch();

export async function startOfflineApi(options: StubOptions = {}): Promise<StubServer> {
  const stub = await startStubServer(options);
  process.env.LEXAPI_BASE_URL = stub.baseUrl;
  process.env.LEXAPI_API_KEY = "lex_offline_test_key";
  assertLocalBaseUrl(process.env.LEXAPI_BASE_URL);
  return stub;
}

/** For tests that build their own client: the environment, checked. */
export function offlineEnv(): NodeJS.ProcessEnv {
  assertLocalBaseUrl(process.env.LEXAPI_BASE_URL);
  return process.env;
}
