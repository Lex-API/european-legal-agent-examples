import OpenAI from "openai";
import { LlmError, providerMessage, type Llm, type LlmRequest } from "./llm.ts";

/** OpenAI Chat Completions with one strict function the model is required to call. */
export class OpenAiLlm implements Llm {
  readonly label: string;
  private readonly client: OpenAI;
  private readonly model: string;

  constructor(options: { model: string; client?: OpenAI; apiKey?: string }) {
    this.model = options.model;
    // Without an apiKey, the SDK reads OPENAI_API_KEY itself.
    this.client = options.client ?? new OpenAI(options.apiKey ? { apiKey: options.apiKey } : {});
    this.label = `openai:${this.model}`;
  }

  async generate(request: LlmRequest): Promise<unknown> {
    const completion = await this.client.chat.completions.create({
      model: this.model,
      messages: [
        { role: "system", content: request.system },
        { role: "user", content: request.user },
      ],
      tools: [
        {
          type: "function",
          function: {
            name: request.tool.name,
            description: request.tool.description,
            parameters: request.tool.schema,
            strict: true,
          },
        },
      ],
      tool_choice: { type: "function", function: { name: request.tool.name } },
    }).catch((error: unknown) => {
      throw new LlmError(providerMessage(error)); // the provider refused or could not be reached
    });

    const message = completion.choices[0]?.message;
    // The refusal text is model-written; it is not echoed into the error.
    if (message?.refusal) throw new LlmError("The model declined this request.");
    const call = message?.tool_calls?.find((c) => c.type === "function" && c.function.name === request.tool.name);
    if (!call || call.type !== "function") throw new LlmError(`The model did not call the ${request.tool.name} function.`);
    try {
      return JSON.parse(call.function.arguments) as unknown;
    } catch {
      throw new LlmError("The model returned function arguments that are not valid JSON.");
    }
  }
}
