import assert from "node:assert/strict";
import { test } from "node:test";
import { McpLegislationSearch } from "../examples/ai-act-provision-finder/src/retrieve-mcp.ts";
import { createLexApiClient } from "./env.ts";

const MISSING = /^LEXAPI_API_KEY is not set\. Create a key at https:\/\/lex-api\.com\/dashboard\?ref=github-examples/;

test("an empty or blank LEXAPI_API_KEY is missing, before any request", async () => {
  for (const value of [undefined, "", "   "]) {
    assert.throws(() => createLexApiClient({ LEXAPI_API_KEY: value }), { message: MISSING });
    await assert.rejects(McpLegislationSearch.start({ LEXAPI_API_KEY: value }), {
      message: "LEXAPI_API_KEY is not set.",
    });
  }
});

test("a key is used without surrounding spaces", () => {
  assert.doesNotThrow(() => createLexApiClient({ LEXAPI_API_KEY: "  lex_offline_test_key  " }));
});
