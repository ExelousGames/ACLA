# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## 5. Product Goal

ACLA is a sim racing telemetry coaching application. Its core value is not only track-specific guidance, but reusable driver development across cars, tracks, sessions, and skill levels.

When designing or modifying product features, prefer functionality that helps users understand and improve their driving behavior over time:
- Persistent driver behavior profiles based on telemetry patterns.
- Progress tracking for consistency, braking, throttle, steering, race pace, and mistake frequency.
- Personalized training plans and practice focus areas.
- Session review summaries that identify strengths, weaknesses, repeated mistakes, and next drills.
- Corner-type coaching that generalizes beyond one track.
- Race stint analysis for consistency, degradation, fatigue, overdriving, and adaptation.

Subscription value should come from long-term coaching, measurable improvement, and reusable insight across many sessions rather than one-time consumption of a track guide.

---

## 6. Application Structure

Main code ownership:
- `acla_front/`: React 19 frontend, Electron packaging scripts, UI views, frontend services, and client-side tests.
- `acla_backend/`: NestJS API on port `7001`. Owns auth, users, racing sessions, AI model metadata, circuit maps, GridFS, MongoDB schemas, and the external `/voice/stream` WebSocket gateway.
- `acla_ai_service/`: Live Python AI/telemetry API on port `8000`. Owns chat, racing engineer logic, voice synthesis/streaming, segment classifier/cropper inference, runtime model hydration, and shared model/telemetry primitives. It must not import the local training package.
- `acla_ai_training/`: Local Streamlit annotation and training workspace on port `8501`. Owns annotation agents/providers, dataset storage, pipelines/manifests, trainer entrypoints, training UI, and local scripts. Imports reusable model/feature code from `acla_ai_service/app`; publishes trained artifacts through the backend active model store.
- `backend_nginx/`: Nginx proxy configs. The frontend talks to this proxy; it forwards HTTP and WebSocket upgrade traffic to `acla_backend_c:7001`.
- `acla_db/`: MongoDB Docker build context, init script, and backup mount.

Compose files:
- `docker-compose.dev.yaml`: Main local development stack: `frontend`, `backend_proxy`, `backend`, `ai_service`, `ai_training`, `mongodb`, and `mongo-express`.
- `docker-compose.prod.yaml`: Production-oriented stack: `frontend`, `backend_proxy`, `backend`, `ai_service`, and `mongodb`.
- `docker-compose.cpu.yaml`, `docker-compose.nvidia.yaml`, `docker-compose.amd.yaml`: AI service and local training hardware/runtime variants (combine with the dev Compose file).

AI container roles:
- `ai_service` runs the live FastAPI server in development and production.
- `ai_training` runs the local annotation/training UI in development; it is absent from the production stack.
- Training owns the existing `ai_models` volume. Serving uses `ai_runtime_models`; model publication crosses the backend API rather than sharing writable trained artifacts.
- Training storage code and pipelines live in `acla_ai_training/training`. Telemetry datasets, backups, and pipeline manifests live in `acla_ai_training/storage`, mounted only by the training container.

Important dev ports:
- Frontend: `${REACT_WEBSITE_PORT}:3000`.
- Backend proxy: `${BACKEND_PROXY_PORT}:80`.
- Backend API: `7001:7001`.
- Live AI service: `8000:8000`.
- Local training UI: `127.0.0.1:8501:8501`.
- MongoDB: `27017:27017`.
- mongo-express: `8081:8081`.

Runtime communication:
- Frontend -> backend proxy: HTTP REST and WebSocket traffic, including `/voice/stream`.
- Backend proxy -> backend: Nginx forwards all paths to `http://acla_backend_c:7001` over `frontend-network`.
- Backend -> MongoDB: NestJS uses Mongoose over `db-network` in dev. Database name is `ACLA`; credentials come from env vars.
- Backend -> AI service: NestJS uses axios for HTTP calls and bridges voice WebSocket traffic to the AI service. The default backend fallback is `AI_SERVICE_URL || http://localhost:8000`.
- Dev AI networking: `backend`, `ai_service`, and `ai_training` share `ai-network`. Both AI containers use `BACKEND_SERVER_IP_FOR_AI` and `BACKEND_PORT` (defaults: `http://backend`, `7001`); the backend uses `AI_SERVICE_URL` (default: `http://ai_service:8000`). Set these in `.dev.env` when the backend and AI containers run on different machines.
- Prod AI networking: `backend` and `ai_service` share `ai-network`. Both AI containers use `BACKEND_SERVER_IP=http://backend` and `BACKEND_PROXY_PORT=7001` for backend access.

Current AI routes include `/health`, `/racing-session/labels`, `/racing-session/top-lap-reference-guidance`, `/racing-session/opportunity-forecast`, `/racing-session/track-corner-knowledge`, `/racing-session/segment-classification`, `/racing-session/live-baseline-analysis`, `/racing-session/analyze-user-sessions`, `/voice/synthesize`, `/voice/voices`, `/voice/health`, `/voice/stream`. Local annotation routes `/annotation/run` and `/annotation/run/stream` are available separately via `training.api.app:app`; they are not mounted on the live service.

The backend AI client still references `/racing-session/train-model` and `/racing-session/train-multiple-models`; verify those routes before relying on them because matching AI service routes were not present in `acla_ai_service/app/api/racing_session.py` when this guide was updated.
