/**
 * The offline rule for tests: nothing may leave this machine.
 *
 * `assertLocalBaseUrl` refuses any LexAPI base URL that is not a 127.0.0.1
 * stub. `blockExternalFetch` replaces the global `fetch` with one that
 * rejects every non-loopback request, so an SDK (LexAPI or a model
 * provider) that slipped through with its default URL fails loudly instead
 * of reaching a real service.
 */
const LOCAL_PREFIX = "http://127.0.0.1:";

export function assertLocalBaseUrl(baseUrl: string | undefined): string {
  if (!baseUrl || !baseUrl.startsWith(LOCAL_PREFIX)) {
    throw new Error(
      `Refusing to run: LEXAPI_BASE_URL must start with ${LOCAL_PREFIX} in tests (got ${baseUrl ? `"${baseUrl}"` : "nothing"}).`,
    );
  }
  return baseUrl;
}

export function blockExternalFetch(): void {
  const realFetch = globalThis.fetch;
  globalThis.fetch = async (input, init) => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    if (!url.startsWith(LOCAL_PREFIX)) {
      throw new Error(`Offline tests blocked a request to ${new URL(url).origin}`);
    }
    return realFetch(input, init);
  };
}
