import { NestFactory } from '@nestjs/core';
import { WsAdapter } from '@nestjs/platform-ws';
import { AppModule } from './app.module';
import { urlencoded, json } from 'express';

async function bootstrap() {
  const app = await NestFactory.create(AppModule, { cors: true });
  // Increase body size limit for large payloads
  app.use(json({ limit: '100mb' }));
  app.use(urlencoded({ extended: true, limit: '100mb' }));

  // Voice WS gateway (VoiceModule → VoiceGateway) needs the `ws` adapter
  // bound to the same underlying HTTP server so the /voice/stream upgrade
  // arrives on the regular backend port. The custom messageParser is a
  // no-op so the framework doesn't JSON.parse every binary PCM frame —
  // we don't use @SubscribeMessage; the gateway handles raw frames.
  app.useWebSocketAdapter(
    new WsAdapter(app, { messageParser: () => undefined }),
  );

  await app.listen(process.env.PORT ?? 7001);
}
bootstrap();
