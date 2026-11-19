/**
 * Retrieval: find AI Act articles by meaning and turn them into `Source`s.
 *
 * One call per query: `semanticLegislationSearch` with `includeText: true`
 * returns the full text of each matched article, so no second request per
 * article is needed (5 credits per query).
 */
import type { LexAPI, SemanticLegislationResult } from "@lexapi/client";
import { creditsCharged } from "../../../shared/env.ts";
import { UnreadableAnswerError } from "../../../shared/lexapi-error.ts";
import { eurLexUrl, type Source } from "../../../shared/grounding.ts";

/** Regulation (EU) 2024/1689 - the Artificial Intelligence Act. */
export const AI_ACT_CELEX = "32024R1689";

/** The fields of a semantic-legislation hit this example reads (same names over REST, the SDK and MCP). */
export type LegislationHit = Pick<
  SemanticLegislationResult,
  "law_id" | "law_title" | "article_ref" | "score" | "text" | "snippet" | "consolidated" | "consolidatedAsOf" | "baseCelex" | "consolidatedNote"
> & {
  /** Set by the MCP route when the tool cut the article text: how many characters are missing. */
  truncatedBy?: number;
};

export interface SearchOutcome {
  hits: LegislationHit[];
  /** Credits the API reported for this call, or `undefined` when the route does not report them. */
  credits: number | undefined;
}

/** How the agent searches. Two implementations: the SDK (below) and the MCP server (`retrieve-mcp.ts`). */
export interface LegislationSearch {
  readonly label: string;
  search(query: string): Promise<SearchOutcome>;
  close(): Promise<void>;
}

export class SdkLegislationSearch implements LegislationSearch {
  readonly label = "@lexapi/client";
  constructor(private readonly client: LexAPI) {}

  async search(query: string): Promise<SearchOutcome> {
    const response = await this.client.semanticLegislationSearch({ query, limit: 10, includeText: true });
    // The SDK returns a 2xx body as it came; an answer without its list of hits is not "no hits".
    if (!isHitList(response?.results)) throw new UnreadableAnswerError(200);
    return { hits: response.results, credits: creditsCharged(response) };
  }

  async close(): Promise<void> {}
}

/** A list of hit objects, as the example reads them. */
export function isHitList(value: unknown): value is LegislationHit[] {
  return Array.isArray(value) && value.every((hit) => hit !== null && typeof hit === "object" && !Array.isArray(hit));
}

/** The act a hit belongs to: a consolidated hit names its base act in `baseCelex`. */
export function actOf(hit: LegislationHit): string | undefined {
  return hit.baseCelex ?? hit.law_id ?? undefined;
}

export interface Retrieved {
  sources: Source[];
  /** Hits that were not AI Act articles (semantic search covers all EU legislation). */
  otherActs: number;
}

/**
 * Keep AI Act hits that have text and an article id, one per article (best
 * score wins), best first, capped at `max`; number them S1, S2, ...
 */
export function toSources(hits: LegislationHit[], max: number): Retrieved {
  const best = new Map<string, LegislationHit>();
  let otherActs = 0;
  for (const hit of hits) {
    if (actOf(hit) !== AI_ACT_CELEX) {
      otherActs++;
      continue;
    }
    if (!hit.article_ref || !hit.text) continue;
    const seen = best.get(hit.article_ref);
    if (!seen || (hit.score ?? 0) > (seen.score ?? 0)) best.set(hit.article_ref, hit);
  }

  const ranked = [...best.values()].sort((a, b) => (b.score ?? 0) - (a.score ?? 0)).slice(0, max);
  const sources = ranked.map((hit, index): Source => {
    const celex = hit.law_id ?? AI_ACT_CELEX;
    const notes: string[] = [];
    if (hit.consolidated) {
      // Consolidated texts have no legal force; say so and name the authentic act.
      notes.push(
        `${hit.consolidatedNote ?? "Consolidated text; not the authentic text."} ` +
          `Authentic act: CELEX ${hit.baseCelex ?? AI_ACT_CELEX} as published in the Official Journal (${eurLexUrl(hit.baseCelex ?? AI_ACT_CELEX)}).`,
      );
    }
    if (hit.truncatedBy) {
      notes.push(`Text truncated by the tool: the last ${hit.truncatedBy} characters of this article were not retrieved, so nothing from them can be quoted.`);
    }
    return {
      id: `S${index + 1}`,
      kind: "legislation",
      celex,
      articleRef: hit.article_ref ?? undefined,
      title: hit.law_title ?? `CELEX ${celex}`,
      url: eurLexUrl(celex),
      text: hit.text ?? "",
      notes,
    };
  });
  return { sources, otherActs };
}
