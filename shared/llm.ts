/**
 * The model call, behind one small interface.
 *
 * Every model call in these examples asks for ONE structured object: the
 * model is given a tool whose input schema is the object we want, and the
 * adapter returns that tool input. Nothing else the model writes is used.
 *
 *   - `ScriptedLlm`  - deterministic, no network. Used by the tests and by `--dry-run`.
 *   - `AnthropicLlm` - Anthropic Messages API with tool use (`llm-anthropic.ts`).
 *   - `OpenAiLlm`    - OpenAI Chat Completions with function calling (`llm-openai.ts`).
 */

export interface LlmTool {
  name: string;
  description: string;
  /** JSON Schema of the object the model must return (strict: every property required, no extras). */
  schema: Record<string, unknown>;
}

export interface LlmRequest {
  system: string;
  user: string;
  tool: LlmTool;
}

export interface Llm {
  /** Human-readable label for logs, e.g. `anthropic:claude-sonnet-5-5`. */
  readonly label: string;
  /** Returns the tool input object produced by the model (unvalidated - callers parse it). */
  generate(request: LlmRequest): Promise<unknown>;
}

export class LlmError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "LlmError";
  }
}

type ScriptedReply = unknown | ((request: LlmRequest) => unknown);

/**
 * A fake model: replies are looked up by tool name. A list is consumed one
 * reply per call (the last one repeats), so a test can script "bad draft,
 * then good draft".
 */
export class ScriptedLlm implements Llm {
  readonly label = "scripted (no model was called)";
  readonly calls: LlmRequest[] = [];
  private readonly replies: Map<string, ScriptedReply[]>;

  constructor(replies: Record<string, ScriptedReply | ScriptedReply[]>) {
    this.replies = new Map(
      Object.entries(replies).map(([tool, reply]) => [tool, Array.isArray(reply) ? [...reply] : [reply]]),
    );
  }

  async generate(request: LlmRequest): Promise<unknown> {
    this.calls.push(request);
    const queue = this.replies.get(request.tool.name);
    if (!queue || queue.length === 0) throw new LlmError(`ScriptedLlm has no reply for tool "${request.tool.name}"`);
    const reply = queue.length > 1 ? queue.shift() : queue[0];
    return typeof reply === "function" ? (reply as (r: LlmRequest) => unknown)(request) : structuredClone(reply);
  }
}

/**
 * Pick the real adapter from the environment:
 *   LLM_PROVIDER=anthropic (default)  ANTHROPIC_API_KEY, ANTHROPIC_MODEL (default claude-sonnet-5-5)
 *   LLM_PROVIDER=openai               OPENAI_API_KEY, OPENAI_MODEL (required)
 */
export async function createLlmFromEnv(env: NodeJS.ProcessEnv = process.env): Promise<Llm> {
  const provider = (env.LLM_PROVIDER ?? "anthropic").toLowerCase();
  if (provider === "anthropic") {
    const { AnthropicLlm } = await import("./llm-anthropic.ts");
    return new AnthropicLlm({ model: env.ANTHROPIC_MODEL });
  }
  if (provider === "openai") {
    if (!env.OPENAI_MODEL) throw new LlmError("Set OPENAI_MODEL to the model id to use (there is no default).");
    const { OpenAiLlm } = await import("./llm-openai.ts");
    return new OpenAiLlm({ model: env.OPENAI_MODEL });
  }
  throw new LlmError(`Unknown LLM_PROVIDER "${provider}" (use "anthropic" or "openai").`);
}
