import Anthropic from "@anthropic-ai/sdk";
import { LlmError, providerMessage, recordingFetch, type Llm, type LlmRequest } from "./llm.ts";

export const DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5-5";

/**
 * Anthropic Messages API with tool use.
 *
 * The tool is declared `strict: true`, so its input is guaranteed to match
 * the schema. `tool_choice` is left at its default (`auto`): current models
 * reject a forced tool choice, so the prompt asks for the tool call and a
 * reply without one is re-prompted once.
 */
export class AnthropicLlm implements Llm {
  readonly label: string;
  private readonly client: Anthropic;
  private readonly model: string;
  /** Keeps the body of the last error answer, as sent. */
  private readonly recorder = recordingFetch();

  constructor(options: { model?: string; client?: Anthropic; apiKey?: string } = {}) {
    this.model = options.model || DEFAULT_ANTHROPIC_MODEL;
    // Without an apiKey, the SDK reads ANTHROPIC_API_KEY itself.
    this.client = options.client ?? new Anthropic({ ...(options.apiKey ? { apiKey: options.apiKey } : {}), fetch: this.recorder.fetch });
    this.label = `anthropic:${this.model}`;
  }

  async generate(request: LlmRequest): Promise<unknown> {
    const messages: Anthropic.MessageParam[] = [
      { role: "user", content: `${request.user}\n\nReply by calling the ${request.tool.name} tool.` },
    ];

    for (let attempt = 0; attempt < 2; attempt++) {
      const response = await this.client.messages
        .create({
          model: this.model,
          max_tokens: 16000,
          system: request.system,
          tools: [
            {
              name: request.tool.name,
              description: request.tool.description,
              input_schema: request.tool.schema as Anthropic.Tool.InputSchema,
              strict: true,
            },
          ],
          messages,
      })
        .catch((error: unknown) => {
          throw new LlmError(providerMessage(error, "ANTHROPIC_API_KEY", this.recorder.lastErrorBody())); // the provider refused or could not be reached
        });

      if (response.stop_reason === "refusal") {
        throw new LlmError("The model declined this request (stop_reason: refusal).");
      }
      if (response.stop_reason === "max_tokens") {
        throw new LlmError("The model's reply was cut off (stop_reason: max_tokens).");
      }
      const toolUse = response.content.find(
        (block): block is Anthropic.ToolUseBlock => block.type === "tool_use" && block.name === request.tool.name,
      );
      if (toolUse) return toolUse.input;

      // No tool call: keep the reply in the conversation and ask again.
      messages.push({ role: "assistant", content: response.content });
      messages.push({ role: "user", content: `Call the ${request.tool.name} tool with your answer.` });
    }
    throw new LlmError(`The model did not call the ${request.tool.name} tool.`);
  }
}
