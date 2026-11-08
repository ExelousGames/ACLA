<p align="center">
  <a href="http://nestjs.com/" target="blank"><img src="https://nestjs.com/img/logo-small.svg" width="120" alt="Nest Logo" /></a>
</p>

[circleci-image]: https://img.shields.io/circleci/build/github/nestjs/nest/master?token=abc123def456
[circleci-url]: https://circleci.com/gh/nestjs/nest

  <p align="center">A progressive <a href="http://nodejs.org" target="_blank">Node.js</a> framework for building efficient and scalable server-side applications.</p>
    <p align="center">
<a href="https://www.npmjs.com/~nestjscore" target="_blank"><img src="https://img.shields.io/npm/v/@nestjs/core.svg" alt="NPM Version" /></a>
<a href="https://www.npmjs.com/~nestjscore" target="_blank"><img src="https://img.shields.io/npm/l/@nestjs/core.svg" alt="Package License" /></a>
<a href="https://www.npmjs.com/~nestjscore" target="_blank"><img src="https://img.shields.io/npm/dm/@nestjs/common.svg" alt="NPM Downloads" /></a>
<a href="https://circleci.com/gh/nestjs/nest" target="_blank"><img src="https://img.shields.io/circleci/build/github/nestjs/nest/master" alt="CircleCI" /></a>
<a href="https://discord.gg/G7Qnnhy" target="_blank"><img src="https://img.shields.io/badge/discord-online-brightgreen.svg" alt="Discord"/></a>
<a href="https://opencollective.com/nest#backer" target="_blank"><img src="https://opencollective.com/nest/backers/badge.svg" alt="Backers on Open Collective" /></a>
<a href="https://opencollective.com/nest#sponsor" target="_blank"><img src="https://opencollective.com/nest/sponsors/badge.svg" alt="Sponsors on Open Collective" /></a>
  <a href="https://paypal.me/kamilmysliwiec" target="_blank"><img src="https://img.shields.io/badge/Donate-PayPal-ff3f59.svg" alt="Donate us"/></a>
    <a href="https://opencollective.com/nest#sponsor"  target="_blank"><img src="https://img.shields.io/badge/Support%20us-Open%20Collective-41B883.svg" alt="Support us"></a>
  <a href="https://twitter.com/nestframework" target="_blank"><img src="https://img.shields.io/twitter/follow/nestframework.svg?style=social&label=Follow" alt="Follow us on Twitter"></a>
</p>
  <!--[![Backers on Open Collective](https://opencollective.com/nest/backers/badge.svg)](https://opencollective.com/nest#backer)
  [![Sponsors on Open Collective](https://opencollective.com/nest/sponsors/badge.svg)](https://opencollective.com/nest#sponsor)-->

## Description

[Nest](https://github.com/nestjs/nest) framework TypeScript starter repository.

Trained Ultralytics model storage: see the [Ultralytics model API contract](src/modules/ai-model/ultralytics/README.md).

## Live Phrase voice APIs

Both endpoints require `Authorization: Bearer <token>` and forward JSON bodies
to the AI service configured by `AI_SERVICE_URL` (default `http://localhost:8000`).
The AI service validates the payloads; its HTTP status and JSON error details
are preserved, including `400` for stale phonemes and `422` for invalid curves.
Connection failures return `503`.

`POST /voice/phonemes` previews the indexed phonemes for a phrase:

```json
{ "text": "Brake smoothly.", "language": "en-us" }
```

The `200` JSON response contains `phonemes`, `tokens` (each with `index`,
`symbol`, and `token_id`), `sample_rate`, and `max_phonemes`. Indices include
spaces, punctuation, and stress markers, and exclude model boundary tokens.

`POST /voice/synthesize/prosody` returns `200 audio/wav` bytes with
`Cache-Control: no-store`:

```json
{
  "text": "Brake smoothly.",
  "voice": "af_bella",
  "speed": 1,
  "language": "en-us",
  "prosody": {
    "expected_phonemes": "<exact phonemes string from the preview>",
    "curves": [
      {
        "start": 0,
        "end": 1,
        "pitch": [{ "position": 0, "value": 0 }, { "position": 1, "value": 2 }]
      }
    ]
  }
}
```

Use the same text and language for preview and synthesis. `language` defaults
to `en-us`; `voice` uses the AI service default and `speed` defaults to `1`
(range `0.5`–`2`). `text` supports 1–4000 characters, subject to the preview's
phoneme limit. `prosody.expected_phonemes` is required; `curves` defaults to an
empty list. Curve ranges are non-overlapping, with an inclusive `start` and
exclusive `end`. Each curve supplies `timing` (duration multiplier `0.25`–`4`),
`energy` (dB `-12`–`12`), or `pitch` (semitones `-12`–`12`), or a combination.
Each control has 2–64 points in increasing `position` order, starting at `0`
and ending at `1`.

## Sessions with video attachments

`GET /racing-session/with-videos` requires an `Authorization: Bearer <token>` header and returns the authenticated user's sessions with at least one MP4 video attachment, newest first. No request body or query parameters are required.

```json
{
  "list": [
    {
      "sessionId": "507f1f77bcf86cd799439011",
      "created_date": "2026-10-13T12:00:00.000Z"
    }
  ]
}
```

`created_date` is the session creation timestamp in ISO 8601 format. If no sessions match, the response is `{ "list": [] }`. Use `GET /racing-session/:id/videos` to list a session's available recordings.

## Migrate stored ACC Y/Z axes

`npm run migrate:acc-yz -- --before <UTC-cutoff>` previews ACC sessions recorded
before the adapter change. Set `MONGODB_URI` (including the database name), or use
the running backend's `MONGO_CLIENTNAME`, `MONGO_CLIENTPASSWORD`, and `MONGO_URL`.
The default dry run inspects metadata only; `--apply` transforms and verifies
every JSON chunk before publishing each session. `--verify` compares every
migrated chunk against its original and fails if eligible sessions remain.

```bash
docker exec acla_backend_c node scripts/migrate-acc-yz.js --before 2026-10-26T16:40:52Z --dry-run
docker exec acla_backend_c node scripts/migrate-acc-yz.js --before 2026-10-26T16:40:52Z --apply
docker exec acla_backend_c node scripts/migrate-acc-yz.js --before 2026-10-26T16:40:52Z --verify
npm run test:migrate:acc-yz
```

The migration swaps all flattened `_y`/`_z` fields and each
`Graphics_car_coordinates` entry. It preserves chunk order, other telemetry,
session metadata, and video attachments. Choose a cutoff preceding the first
normalized capture; never extend it to include new recordings. Legacy sessions
without `created_date` use their ObjectId timestamp, and source chunk upload
dates must also precede the cutoff.

Each session atomically receives its replacement file IDs and a
`telemetryMigrations.acc_yz_v1` marker, so reruns skip completed sessions.
Original GridFS files remain available through the marker's
`originalChunkFileIds` for recovery. Allow space for a second copy of the selected
telemetry. Interrupted sessions keep their original references; rerunning can
leave unused staged copies identified by `metadata.telemetryMigration` in
`racing_sessions.files`. No original or staged files are automatically deleted.
To recover a session, restore its ordered `originalChunkFileIds` and remove the
marker in one conditional update after checking its current references.

## Repair stored ACC orientation after the Y/Z migration

The follow-up `acc_orientation_v1` migration negates `Physics_heading`,
`Physics_pitch`, `Physics_roll`, and all three `Physics_local_angular_vel_*`
components. Swapping Y/Z has determinant -1, so right-hand rotations require
these sign changes. Positions and polar vectors are already normalized and
are preserved. This matches the updated frontend ACC adapter.

```bash
docker exec acla_backend_c node scripts/migrate-acc-orientation.js --before 2026-10-31T01:37:46Z --dry-run
docker exec acla_backend_c node scripts/migrate-acc-orientation.js --before 2026-10-31T01:37:46Z --apply
docker exec acla_backend_c node scripts/migrate-acc-orientation.js --before 2026-10-31T01:37:46Z --verify
npm run test:migrate:acc-orientation
```

The cutoff precedes captures from the orientation-aware adapter. Every candidate
must have a completed `acc_yz_v1` marker, and each current chunk must trace to
that migration in the original order. Unmarked sessions are rejected because
their coordinate convention cannot be inferred safely. The default dry run
checks metadata only. Apply uses the same verified-copy and atomic-publish path
as the Y/Z migration, retaining both the raw originals and the Y/Z-only copies.
The new marker prevents repeat sign changes. Verify checks every repaired chunk
against its Y/Z-only backup; use this follow-up verifier after repair, since the
historical Y/Z verifier expects the intermediate representation.

To roll back only this repair, restore the ordered `originalChunkFileIds` from
`telemetryMigrations.acc_orientation_v1` and remove that marker in a single
conditional session update after checking current references. Keep `acc_yz_v1`
and its original files intact. No files are deleted by either migration.

## Project setup

```bash
$ npm install
```

## Compile and run the project

```bash
# development
$ npm run start

# watch mode
$ npm run start:dev

# production mode
$ npm run start:prod
```

## Run tests

```bash
# unit tests
$ npm run test

# e2e tests
$ npm run test:e2e

# test coverage
$ npm run test:cov
```

## Deployment

When you're ready to deploy your NestJS application to production, there are some key steps you can take to ensure it runs as efficiently as possible. Check out the [deployment documentation](https://docs.nestjs.com/deployment) for more information.

If you are looking for a cloud-based platform to deploy your NestJS application, check out [Mau](https://mau.nestjs.com), our official platform for deploying NestJS applications on AWS. Mau makes deployment straightforward and fast, requiring just a few simple steps:

```bash
$ npm install -g @nestjs/mau
$ mau deploy
```

With Mau, you can deploy your application in just a few clicks, allowing you to focus on building features rather than managing infrastructure.

## Resources

Check out a few resources that may come in handy when working with NestJS:

- Visit the [NestJS Documentation](https://docs.nestjs.com) to learn more about the framework.
- For questions and support, please visit our [Discord channel](https://discord.gg/G7Qnnhy).
- To dive deeper and get more hands-on experience, check out our official video [courses](https://courses.nestjs.com/).
- Deploy your application to AWS with the help of [NestJS Mau](https://mau.nestjs.com) in just a few clicks.
- Visualize your application graph and interact with the NestJS application in real-time using [NestJS Devtools](https://devtools.nestjs.com).
- Need help with your project (part-time to full-time)? Check out our official [enterprise support](https://enterprise.nestjs.com).
- To stay in the loop and get updates, follow us on [X](https://x.com/nestframework) and [LinkedIn](https://linkedin.com/company/nestjs).
- Looking for a job, or have a job to offer? Check out our official [Jobs board](https://jobs.nestjs.com).

## Support

Nest is an MIT-licensed open source project. It can grow thanks to the sponsors and support by the amazing backers. If you'd like to join them, please [read more here](https://docs.nestjs.com/support).

## Stay in touch

- Author - [Kamil Myśliwiec](https://twitter.com/kammysliwiec)
- Website - [https://nestjs.com](https://nestjs.com/)
- Twitter - [@nestframework](https://twitter.com/nestframework)

## License

Nest is [MIT licensed](https://github.com/nestjs/nest/blob/master/LICENSE).
