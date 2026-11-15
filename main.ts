// free-model-aggregator-02 — one server, two client wires, the best free model behind every answer.
//
// A single OpenRouter lane: every prompt is dialled on the grader's gateway (Chat Completions) on the
// entrant's OpenRouter key, as the brief's `id` spells it. Routing is by the brief's own quality
// rating at the prompt's requirements, with failover across every free model that meets them and a
// refusal trail on the response, so a refused call never loses a prompt it could still serve.
//
// What this entry does differently from the house baseline:
//   - quality: candidates are ranked by rating_milli (highest first), not by brief order, so the
//     strongest model that can serve the prompt is always tried first;
//   - failover: a 4xx/5xx/network refusal moves to the next ranked candidate instead of ending the
//     prompt, and every refusal is reported on the x-makefun-upstream-failures header;
//   - rate limits: a model that refused is never retried on the same prompt, and within one rating
//     band refused models sort after ones that never refused, so a cooling model costs no calls;
//   - streams are validated before the first byte reaches the client: a first frame carrying an
//     error fails over like any other refusal, instead of relaying an error as an answer.
//
// Everything a route returns is a function of the requests and the gateway's answers seen so far:
// no wall clock, no randomness. Model lists use fixed timestamps, so --list, the register body and
// every GET /v1/models answer reproduce byte for byte on a replay.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** One gateway attempt may take this long before the entry fails over to the next model. */
const ATTEMPT_TIMEOUT_MS = 20000;
/** The output cap a request without one is given, and the most the gateway accepts. */
const DEFAULT_MAX_TOKENS = 1024;
const MAX_TOKENS_CAP = 8192;
/** At most this many refusals are reported on the x-makefun-upstream-failures header. */
const MAX_FAILURE_ROWS = 8;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const MODELS_SCHEMA = 'free-model-aggregator-02/models/v1';
const PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1';
/** The public base URL of the provider the free models belong to: the endpoint the gateway reaches. */
const OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1';
/** The environment variable an entrant's OpenRouter key is pushed under (an entry secret). */
const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';
/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

/** One free model as the brief lists it. `id` is the id the gateway routes. */
interface ModelRow {
  id: string;
  /** The provider the model belongs to, as the brief names it (`openrouter`). */
  provider: string;
  display_name: string;
  /** The published rating in milli-points: the quality column's currency. */
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

/** What one prompt needs: a model without it cannot serve the prompt. */
interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

/** One prompt the client will send, as the brief lists it. */
interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

/** The brief as this entry reads it. `error` is set when a document could not be read. */
interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

/** One model in the catalog document. */
interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** One provider in the catalog document. */
interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: 'openai';
  models: CatalogModel[];
}

/** What `--list` prints and what the entry POSTs to /register: the same document, byte for byte. */
interface CatalogReport {
  providers: CatalogProvider[];
}

/** What a model can do, from the brief's row (null: the brief did not say). */
interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** One refused gateway call, as the x-makefun-upstream-failures header reports it. */
interface Failure {
  status: number;
  code: string;
  model: string | null;
}

/**
 * The JSON document in `text` whose `schema` member is `schema`, parsed. The brief writes each
 * document as one object that opens with its schema, so the document is found by the schema string
 * itself: the brace just before it opens the object, and the matching brace closes it. String
 * contents are skipped while the braces are counted, so a brace inside a string never ends the doc.
 */
function docWithSchema(text: string, schema: string): Json | null {
  const marker = text.indexOf(schema);
  if (marker < 0) return null;
  const open = text.lastIndexOf('{', marker);
  if (open < 0) return null;
  if (!/^\s*"schema"\s*:\s*"$/.test(text.slice(open + 1, marker))) return null;
  let depth = 0;
  let quoted = false;
  for (let i = open; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '\\') i += 1;
      else if (ch === '"') quoted = false;
      continue;
    }
    if (ch === '"') {
      quoted = true;
      continue;
    }
    if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) {
        try {
          const parsed: unknown = JSON.parse(text.slice(open, i + 1));
          return isRecord(parsed) ? parsed : null;
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}

/**
 * One model row, normalised. The id comes from the row's `id`, or its `model` under the provider
 * that names it: a bare OpenRouter id (`vendor/model:free`) naming `openrouter` is routed as the
 * gateway spells it (`openrouter/vendor/model:free`). The brief lists free models, so a row is free
 * unless it says `free: false`; a paid row is never routed.
 */
function asModelRow(v: unknown): ModelRow | null {
  if (!isRecord(v)) return null;
  const raw = typeof v.id === 'string' && v.id !== '' ? v.id : typeof v.model === 'string' ? v.model : '';
  if (raw === '') return null;
  const providerName =
    typeof v.provider === 'string' && v.provider !== '' ? v.provider : typeof v.wire === 'string' ? v.wire : 'openrouter';
  const id = providerName === 'openrouter' && !raw.toLowerCase().startsWith('openrouter/') ? `openrouter/${raw}` : raw;
  const num = (x: unknown): number | null => (typeof x === 'number' && Number.isFinite(x) ? x : null);
  const flag = (x: unknown): boolean | null => (typeof x === 'boolean' ? x : null);
  return {
    id,
    provider: providerName,
    display_name: typeof v.display_name === 'string' && v.display_name !== '' ? v.display_name : raw,
    rating_milli: num(v.rating_milli) ?? 0,
    context_length: num(v.context_length),
    tools: flag(v.tools),
    vision: flag(v.vision),
    json: flag(v.json),
    free: v.free !== false,
  };
}

/** One prompt row, normalised: only the requirements the brief actually sets are kept. */
function asPromptRow(v: unknown): PromptRow | null {
  if (!isRecord(v)) return null;
  if (typeof v.id !== 'string' || v.id === '') return null;
  const req = isRecord(v.requires) ? v.requires : {};
  const needs: Requirement = {};
  if (req.tools === true) needs.tools = true;
  if (req.vision === true) needs.vision = true;
  if (req.json === true) needs.json = true;
  if (typeof req.min_context === 'number' && Number.isFinite(req.min_context)) needs.min_context = req.min_context;
  return {
    id: v.id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: needs,
  };
}

/** The brief at `path`: the free model list and the six prompts, each found by its own schema. */
function readBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const modelsDoc = docWithSchema(text, MODELS_SCHEMA);
  if (modelsDoc === null) {
    return { models: [], prompts: [], error: `the brief at ${path} holds no ${MODELS_SCHEMA} document` };
  }
  // A brief may spell the model list `models` or `ratings` (the sibling contest's name).
  const rows = Array.isArray(modelsDoc.models) ? modelsDoc.models : Array.isArray(modelsDoc.ratings) ? modelsDoc.ratings : [];
  const models: ModelRow[] = [];
  for (const row of rows) {
    const m = asModelRow(row);
    if (m !== null) models.push(m);
  }
  const promptsDoc = docWithSchema(text, PROMPTS_SCHEMA);
  if (promptsDoc === null) {
    return { models, prompts: [], error: `the brief at ${path} holds no ${PROMPTS_SCHEMA} document` };
  }
  // A brief may spell the prompt list `prompts` or `manifest` (the sibling contest's name).
  const prows = Array.isArray(promptsDoc.prompts) ? promptsDoc.prompts : Array.isArray(promptsDoc.manifest) ? promptsDoc.manifest : [];
  const prompts: PromptRow[] = [];
  for (const row of prows) {
    const p = asPromptRow(row);
    if (p !== null) prompts.push(p);
  }
  return { models, prompts, error: null };
}

/** The models this entry may serve: the brief's free ones, in the brief's order. */
function freeModels(brief: Brief): ModelRow[] {
  return brief.models.filter((m) => m.free);
}

/** What a model can do, as the brief's row says it. */
function capabilitiesOf(model: ModelRow): Capabilities {
  return {
    context_length: model.context_length,
    tools: model.tools,
    vision: model.vision,
    json: model.json,
  };
}

/** Whether a model's capabilities meet one prompt's requirements. An unknown capability never meets one. */
function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

/**
 * The catalog the brief's free model list makes: one provider, the models it serves. The base_url is
 * the provider's public endpoint, never the gateway's trial-local address: the gateway's origin differs
 * between a live pass and its replay, and the catalog is part of the state export a replay must
 * reproduce byte for byte. Every free model the brief names is listed, exactly once.
 */
function buildCatalog(brief: Brief): CatalogReport {
  const seen = new Set<string>();
  const models: CatalogModel[] = [];
  for (const model of brief.models) {
    if (!model.free || seen.has(model.id)) continue;
    seen.add(model.id);
    models.push({ id: model.id, free: true, context_length: model.context_length });
  }
  if (models.length === 0) return { providers: [] };
  return {
    providers: [
      {
        id: 'openrouter',
        base_url: OPENROUTER_PUBLIC_BASE,
        key_env: OPENROUTER_KEY_ENV,
        wire: 'openai',
        models,
      },
    ],
  };
}

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;
/** How many times each model has refused a gateway call, across the prompts served so far. */
const refusalCount = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/* ------------------------------------------------------------------ */
/* The client wires: model lists, error envelopes, Anthropic <-> OpenAI */
/* ------------------------------------------------------------------ */

/** One model the entry serves, as GET /v1/models lists it. */
interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text carried by a message body: a plain string, or its text blocks joined. */
function blockText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  let out = '';
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') {
      out += (out === '' ? '' : '\n') + block.text;
    }
  }
  return out;
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/**
 * An Anthropic Messages request as the Chat Completions request the gateway routes, naming `model`:
 * the system prompt becomes the first system message, image blocks become image_url parts, tool_use
 * blocks become tool_calls on the assistant message, tool_result blocks become tool messages, and
 * tools, tool_choice, the output cap, temperature, top_p, stop sequences and streaming carry over.
 */
function anthropicRequestToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = blockText(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: raw.role === 'assistant' ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (raw.role === 'assistant') {
      const calls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        calls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = blockText(content);
      const msg: Json = { role: 'assistant', content: text === '' ? null : text };
      if (calls.length > 0) msg.tool_calls = calls;
      messages.push(msg);
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        const inner = block.content;
        let text: string;
        if (typeof inner === 'string') text = inner;
        else if (Array.isArray(inner)) {
          const t = blockText(inner);
          text = t === '' ? JSON.stringify(inner) : t;
        } else text = inner === undefined ? '' : JSON.stringify(inner);
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: text,
        });
        continue;
      }
      if (block.type === 'text' && typeof block.text === 'string') {
        parts.push({ type: 'text', text: block.text });
        continue;
      }
      if (block.type === 'image' && isRecord(block.source)) {
        const src = block.source;
        if (src.type === 'base64' && typeof src.data === 'string') {
          parts.push({
            type: 'image_url',
            image_url: { url: `data:${typeof src.media_type === 'string' ? src.media_type : 'image/png'};base64,${src.data}` },
          });
        } else if (src.type === 'url' && typeof src.url === 'string') {
          parts.push({ type: 'image_url', image_url: { url: src.url } });
        }
      }
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools: Json[] = [];
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        type: 'function',
        function: {
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        },
      });
    }
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A Chat Completions answer as an Anthropic Messages object. `model` is the id the entry dialled. */
function openAIResponseToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : blockText(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    let input: Json = {};
    const args = call.function.arguments;
    if (isRecord(args)) input = args;
    else if (typeof args === 'string' && args.trim() !== '') {
      try {
        const parsed: unknown = JSON.parse(args);
        if (isRecord(parsed)) input = parsed;
      } catch {
        // Not JSON: an empty input.
      }
    }
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input,
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** A Chat Completions answer with `model` set to the id the entry dialled: the one truthful claim. */
function openAIJsonWithModel(body: Json, model: string): Json {
  return { ...body, model };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * One SSE frame with every data payload's `model` set to `model`, and its blank line kept: comments,
 * `event:` lines and payloads that are not JSON objects pass through untouched. Chat Completions names
 * the model on its chunks; the client sees the id the entry dialled, never the provider's spelling.
 */
function frameWithModel(frame: string, model: string): string {
  const lines = frame.split('\n').map((line) => {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) return line;
    const payload = trimmed.slice(5).trim();
    if (payload === '' || payload === '[DONE]') return line;
    try {
      const chunk: unknown = JSON.parse(payload);
      if (isRecord(chunk)) return `data: ${JSON.stringify({ ...chunk, model })}`;
    } catch {
      // Not JSON: relay the line as it came.
    }
    return line;
  });
  return `${lines.join('\n')}\n\n`;
}

/**
 * A Chat Completions SSE body as complete frames, split on the blank line that ends them, as the SSE
 * grammar does; a frame cut across reads is held until its end arrives, and a body that ends without
 * one is relayed as it is.
 */
async function* sseFrames(upstream: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        yield frame;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') yield buffer;
  } finally {
    reader.releaseLock();
  }
}

/** The data payloads of one frame, parsed as JSON where they parse. */
function framePayloads(frame: string): unknown[] {
  const out: unknown[] = [];
  for (const line of frame.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) continue;
    const payload = trimmed.slice(5).trim();
    if (payload === '' || payload === '[DONE]') continue;
    try {
      out.push(JSON.parse(payload));
    } catch {
      // Not JSON: not a completion chunk.
    }
  }
  return out;
}

/** Whether a frame carries a gateway error object (a refusal inside a 200 stream), or null. */
function frameError(frame: string): string | null {
  for (const payload of framePayloads(frame)) {
    if (isRecord(payload) && isRecord(payload.error)) return typeof payload.error.message === 'string' ? payload.error.message : 'gateway error';
  }
  return null;
}

/** Whether a frame carries at least one completion chunk (a choices array). */
function frameHasChunk(frame: string): boolean {
  for (const payload of framePayloads(frame)) {
    if (isRecord(payload) && Array.isArray(payload.choices)) return true;
  }
  return false;
}

/* -------------------------------------------------------------- */
/* Stream relays: validated first frame, then the rest as it comes */
/* -------------------------------------------------------------- */

/** The refusal trail, when the answer follows at least one refused call; empty otherwise. */
function failureHeader(failures: readonly Failure[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, MAX_FAILURE_ROWS)) };
}

/** A Chat Completions SSE stream as it came, frame by frame, every data payload's `model` the id dialled. */
async function relayOpenAIStream(
  res: ServerResponse,
  model: string,
  failures: readonly Failure[],
  firstFrame: string,
  frames: AsyncGenerator<string>,
): Promise<void> {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...failureHeader(failures) });
  res.write(frameWithModel(firstFrame, model));
  for await (const frame of frames) res.write(frameWithModel(frame, model));
  res.end();
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason
 * and usage, then message_stop. `model` is the id the entry dialled, named back to the client.
 */
async function relayAnthropicStream(
  res: ServerResponse,
  model: string,
  failures: readonly Failure[],
  firstFrame: string,
  frames: AsyncGenerator<string>,
): Promise<void> {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...failureHeader(failures) });
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;
  /** Whether any text or tool input was produced: the floor for the output count when no usage arrived. */
  let producedAny = false;

  const startEvents = (): void => {
    if (started) return;
    started = true;
    res.write(
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    );
  };
  const closeEvents = (): void => {
    if (textIndex !== -1) {
      res.write(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) {
      res.write(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    }
    toolIndexes.clear();
  };
  const finishEvents = (stopReason: string): void => {
    if (finished) return;
    finished = true;
    closeEvents();
    // The wire wants an output count of at least 1 when the model produced anything; the usage chunk
    // normally carries it, and this is only the floor when it never arrived.
    const output = outputTokens > 0 ? outputTokens : producedAny ? 1 : 0;
    res.write(
      sseEvent('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: stopReason, stop_sequence: null },
        usage: { output_tokens: output },
      }),
    );
    res.write(sseEvent('message_stop', { type: 'message_stop' }));
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): void => {
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        startEvents();
        finishEvents(pendingStop ?? 'end_turn');
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      startEvents();
      if (typeof delta.content === 'string' && delta.content !== '') {
        producedAny = true;
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          res.write(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        res.write(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            res.write(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          res.write(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          producedAny = true;
          res.write(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived:
      // OpenAI sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
  };

  eventsOfFrame(firstFrame);
  for await (const frame of frames) eventsOfFrame(frame);
  if (!finished) {
    startEvents();
    finishEvents(pendingStop ?? 'end_turn');
  }
  res.end();
}

/* ---------------------------------------------------------- */
/* The server: routes, rating-ranked routing, failover, poll  */
/* ---------------------------------------------------------- */

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, failures: readonly Failure[] = []): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...failureHeader(failures) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, failures: readonly Failure[] = []): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), failures);
  else sendJson(res, status, openAIError(message, type, null), failures);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), {
    method: 'POST',
    headers,
    body: JSON.stringify({ ...body, model }),
    signal: AbortSignal.timeout(ATTEMPT_TIMEOUT_MS),
  });
}

/** The gateway's error code for a refused call: its own code, its type, or its HTTP status. */
function failureCodeOfBody(text: string, status: number): string {
  try {
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed) && isRecord(parsed.error)) {
      if (typeof parsed.error.code === 'string' && parsed.error.code !== '') return parsed.error.code;
      if (typeof parsed.error.type === 'string' && parsed.error.type !== '') return parsed.error.type;
    }
  } catch {
    // Not JSON: fall through to the status.
  }
  return `http_${status}`;
}

/** The code for a call that never reached the gateway: a timeout, or an unreachable upstream. */
function failureCodeOfError(err: unknown): string {
  const name = err instanceof Error ? err.name : '';
  if (name === 'TimeoutError' || name === 'AbortError') return 'timeout';
  return 'upstream_unreachable';
}

/**
 * The free models that can serve a request, best first: the brief's rating_milli, highest first; a
 * refused model sorts after one that never refused inside the same rating band, so a cooling model
 * costs no call; the brief's order breaks the rest. A model that refused is never retried on the same
 * prompt — the failover loop tries each candidate once.
 */
function rankCandidates(requires: Requirement): ModelRow[] {
  return freeModels(brief)
    .map((model, briefIndex) => ({ model, briefIndex }))
    .filter(({ model }) => meetsRequirements(requires, capabilitiesOf(model)))
    .sort(
      (a, b) =>
        b.model.rating_milli - a.model.rating_milli ||
        (refusalCount.get(a.model.id) ?? 0) - (refusalCount.get(b.model.id) ?? 0) ||
        a.briefIndex - b.briefIndex,
    )
    .map(({ model }) => model);
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json, wire: 'openai' | 'anthropic', stream: boolean): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  // The manifest is the declared source, but only when it describes the request that actually
  // arrived; a row that names another wire or another streaming mode is not this prompt's row.
  if (entry !== undefined && entry.api === wire && entry.stream === stream) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/**
 * The client body as the gateway's Chat Completions body for one model: an OpenAI-shaped body carries
 * its messages, system prompt, tools, max_tokens and temperature through unchanged, an Anthropic-shaped
 * one is translated into that body first. A request without an output cap gets one, and no cap the
 * gateway accepts is ever exceeded; a streamed request asks for its usage on the last chunk.
 */
function buildUpstreamBody(body: Json, wire: 'openai' | 'anthropic', model: string): Json {
  const out = wire === 'anthropic' ? anthropicRequestToOpenAI(body, model) : { ...body, model };
  if (typeof out.max_tokens !== 'number' && typeof out.max_completion_tokens !== 'number') out.max_tokens = DEFAULT_MAX_TOKENS;
  if (typeof out.max_tokens === 'number') out.max_tokens = Math.min(out.max_tokens, MAX_TOKENS_CAP);
  if (typeof out.max_completion_tokens === 'number') out.max_completion_tokens = Math.min(out.max_completion_tokens, MAX_TOKENS_CAP);
  if (out.stream === true) out.stream_options = { include_usage: true };
  return out;
}

/**
 * One completion, on either wire: try the ranked candidates in order, fail over on every refusal, and
 * serve the first model that answers. A 401 stops the loop — the key is broken and every further call
 * refuses the same way. When every candidate refused, the prompt is answered 503 with the refusal
 * trail, never with a made-up completion.
 */
async function handleCompletion(req: IncomingMessage, res: ServerResponse, wire: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, wire, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const candidates = rankCandidates(requirementsFor(body, wire, stream));
  if (candidates.length === 0) return sendError(res, wire, 503, 'no_free_model', 'no free model in the brief meets this request');
  const failures: Failure[] = [];
  for (const candidate of candidates) {
    const upstreamBody = buildUpstreamBody(body, wire, candidate.id);
    let upstream: Response;
    try {
      upstream = await dial(candidate.id, upstreamBody);
    } catch (err) {
      refusalCount.set(candidate.id, (refusalCount.get(candidate.id) ?? 0) + 1);
      failures.push({ status: 0, code: failureCodeOfError(err), model: candidate.id });
      continue;
    }
    if (!upstream.ok || upstream.body === null) {
      const text = await upstream.text().catch(() => '');
      refusalCount.set(candidate.id, (refusalCount.get(candidate.id) ?? 0) + 1);
      failures.push({ status: upstream.status, code: failureCodeOfBody(text, upstream.status), model: candidate.id });
      if (upstream.status === 401) break;
      continue;
    }
    if (stream) {
      const frames = sseFrames(upstream.body);
      const first = await frames.next();
      if (first.done) {
        refusalCount.set(candidate.id, (refusalCount.get(candidate.id) ?? 0) + 1);
        failures.push({ status: 200, code: 'empty_stream', model: candidate.id });
        continue;
      }
      if (frameError(first.value) !== null) {
        refusalCount.set(candidate.id, (refusalCount.get(candidate.id) ?? 0) + 1);
        failures.push({ status: 200, code: 'stream_error', model: candidate.id });
        continue;
      }
      if (wire === 'openai') await relayOpenAIStream(res, candidate.id, failures, first.value, frames);
      else await relayAnthropicStream(res, candidate.id, failures, first.value, frames);
      return;
    }
    const answer: unknown = await upstream.json().catch(() => null);
    if (!isRecord(answer) || isRecord(answer.error)) {
      refusalCount.set(candidate.id, (refusalCount.get(candidate.id) ?? 0) + 1);
      failures.push({ status: 200, code: isRecord(answer) ? failureCodeOfBody(JSON.stringify(answer), 200) : 'bad_json', model: candidate.id });
      continue;
    }
    if (wire === 'openai') sendJson(res, 200, openAIJsonWithModel(answer, candidate.id), failures);
    else sendJson(res, 200, openAIResponseToAnthropic(answer, candidate.id), failures);
    return;
  }
  return sendError(res, wire, 503, 'all_models_refused', 'every free model that meets this request refused the gateway call', failures);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await handleCompletion(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await handleCompletion(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  let lastErr: unknown = null;
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
      return;
    } catch (err) {
      lastErr = err;
      await sleep(500);
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error(String(lastErr));
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
