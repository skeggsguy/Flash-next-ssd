# The chat window

One server, two users: a person chatting, and a coding agent working in the background. On a 64 GB
Mac there is memory for only one conversation at a time (`-np 1`), so the two take turns. The chat
window decides the turns: **chat goes first**.

<p align="center"><img src="../../media/flashnext/chat-window.svg" width="760" alt="The chat window: a chat request cuts a running coding request off, its retry is held while the window is open, and coding resumes when the window lapses"></p>

## Turning it on

```sh
llama-server ... -np 1 --chat-window 300
```

| Option | Env var | Default | What it does |
|---|---|---|---|
| `--chat-window N` | `LLAMA_ARG_CHAT_WINDOW` | `0` (off) | Seconds after the last chat reply during which background requests wait. The study uses 300 (5 minutes). |
| `--chat-window-header NAME` | `LLAMA_ARG_CHAT_WINDOW_HEADER` | `X-Lane` | The request header that marks background requests. |
| `--chat-window-background VALUE` | `LLAMA_ARG_CHAT_WINDOW_BACKGROUND` | `code` | The header value that means "background". |
| `--sse-ping-interval N` | `LLAMA_ARG_SSE_PING_INTERVAL` | `30` | How often a waiting streamed request gets a keep-alive comment. |

In a `--models-preset` INI file:

```ini
[flashnext]
parallel = 1
chat-window = 300
chat-window-header = X-Lane
chat-window-background = code
```

## How it behaves

- Requests with `X-Lane: code` are **background** requests. Every other request, with or without
  the header, is **chat**. So only the coding agent needs configuring; your chat client needs nothing.
  The header name and value match ignoring case.
- **A chat request cuts a running background request off** at the next batch and runs instead. The
  background request ends with a `503` error of type `overloaded_error`. If it was streaming, the
  error arrives as an SSE `error` event; on `/v1/messages` it has Anthropic's error shape. Clients
  that retry on overloaded or 5xx errors simply send it again.
- **Background requests wait** while chat runs and for `--chat-window` seconds after the last chat
  reply. Every chat reply restarts the clock. When the window lapses, the waiting requests start by
  themselves, oldest first.
- **A waiting streamed request is kept alive**: the server sends the HTTP headers and then an SSE
  comment every `--sse-ping-interval` seconds, so the client does not time out.
- It covers the completion endpoints: `/completion`, `/infill`, `/v1/completions`,
  `/v1/chat/completions`, `/v1/responses` and `/v1/messages`. Embeddings, tokenize and the rest are
  never cut or held.

The log shows a `chat window:` line at startup and for each event: `cut code task N for chat task M`,
`held code task N for S s more` and `window lapsed`.

## Marking the coding agent

With curl:

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Lane: code" \
  -d '{"messages": [{"role": "user", "content": "Refactor this function"}], "stream": true}'
```

In an [OpenCode](https://opencode.ai) provider (`opencode.json`):

```json
{
  "provider": {
    "llama-server": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "headers": { "X-Lane": "code" }
      },
      "models": { "local": { "name": "Flash-Next" } }
    }
  }
}
```

The study checked that OpenCode retries a cut request by itself, through both the OpenAI-style and
the Anthropic-style endpoint. A separate live check measured the cut 5.6 s after the chat request
arrived and the held request starting 60.7 s after the chat reply (with a 60 s window), and the
words were identical to a run with the window off.

## Things to know

- **A client that forgets the header counts as chat.** If your coding agent is not marked, it will
  cut your chat off instead of the other way round.
- **Use it with `-np 1`.** With more slots both sides could run at once anyway, and on 64 GB there is
  no memory for more slots at a 200K context.
- **Non-streamed background requests get no keep-alive.** They hear nothing until they run, which
  can be the whole window. Give that client a long timeout, or have it stream.
- **A switch costs a prompt read.** Each side keeps its cached prompt only while it holds the seat,
  so after a switch the first request reads its prompt in again (unless `--cache-ram` still holds
  it; the everyday command sets it to 0 to give that memory to the expert cache). On a long coding
  conversation that can take a minute or more.
- **The header is a scheduling hint, not access control.** Anyone who can reach the server can send
  any header. Keep the server on `127.0.0.1`.
