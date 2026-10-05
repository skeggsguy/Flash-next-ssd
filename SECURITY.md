# Security

Flash-Next SSD is a personal fork of [llama.cpp](https://github.com/ggml-org/llama.cpp), maintained
on a best-effort basis.

## Reporting a problem

- **In this fork's own code** (expert streaming, the reading room, the chat window and the other
  parts listed in [AGENTS.md](AGENTS.md)): please report it privately through a
  [security advisory](https://github.com/skeggsguy/Flash-next-ssd/security/advisories/new) on this
  repository, not as a public issue.
- **In upstream llama.cpp code**: report it to upstream, following
  [their security policy](https://github.com/ggml-org/llama.cpp/blob/master/SECURITY.md). Fixes
  there reach this fork when it merges upstream.

## Using it safely

Upstream's guidance on [using llama.cpp securely](https://github.com/ggml-org/llama.cpp/blob/master/SECURITY.md#using-llamacpp-securely)
applies here too: treat model files and prompts from others as untrusted, and do not expose the
server to networks you do not trust. In particular:

- The commands in these guides bind the server to `127.0.0.1`. Keep it that way unless you add
  your own access control in front of it.
- The chat window's header (`X-Lane: code`) is a scheduling hint, not access control: any client
  that can reach the server can send it.
