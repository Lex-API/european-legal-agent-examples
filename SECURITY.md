# Security

## API keys and secrets

- Keys are read from environment variables only (`LEXAPI_API_KEY`, `ANTHROPIC_API_KEY`,
  `OPENAI_API_KEY`, `LEXAPI_WEBHOOK_SECRET`). Each example has a `.env.example`; copy it to `.env`,
  which is git-ignored. Never commit a key, and never put one in client-side code.
- A LexAPI key can be rotated or revoked from the dashboard. If a key was exposed, revoke it.
- The webhook secret is returned once, when the webhook is created. Store it like a key.

## Webhook deliveries

The change monitor's receiver verifies every delivery before using it, with the SDK's
`verifyWebhook` helper: HMAC-SHA256 of the raw request body keyed with the webhook secret,
compared in constant time, plus a 5-minute window on the signed `timestamp`. It also drops
deliveries whose `delivery.id` it has already handled. The signature covers the body only, so the
receiver does not trust request headers such as `X-Webhook-ID`.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Email support@lex-api.com with the details.
