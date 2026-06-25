# Voice tool knowledge

One markdown file per voice tool. These files are the AI service's source
of truth for LLM-facing tool titles, descriptions, and parameter wording.

Executable handlers stay in code:

- Server tools run inside the AI service.
- Frontend tools are relayed to the Electron/browser client over the voice
  websocket.

Frontend handshakes should provide only tool names and JSON parameter
shapes. The voice pipeline enriches those capabilities with this corpus
before exposing tools to the LLM.

