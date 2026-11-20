# Managed Slack OAuth service

The broker runs separately from Xerxes Hub. It keeps the Slack app secret on the broker and hands a workspace user token to the initiating Hub over HTTPS. It does not store Hub identities, send Slack messages, or persist tokens. Run only on a trusted operator-controlled host; the broker necessarily sees the issued Slack token.

## Operator setup

1. Create a Slack app. Add **User Token Scopes** `users:read`, `users:read.email`, `im:write`, `im:read`, `im:history`, and `chat:write`. Do not add bot scopes or enable token rotation or organization-wide installation. Admin approval may be required in the intended workspace.
2. Register exactly `https://<broker-host>/v1/slack/callback` as the Slack redirect URL. Use a dedicated HTTPS origin (not a path on the Hub or a shared website).
3. Put the app credentials in the broker process environment via your secret manager. From `xerxes/`, the entry point is `bun src/slack-broker/server.ts`. Configuration:

   | Variable | Value |
   | --- | --- |
   | `XERXES_SLACK_BROKER_ORIGIN` | Public HTTPS origin, e.g. `https://slack-connect.example.org` |
   | `XERXES_SLACK_BROKER_CLIENT_ID` | Slack app client ID |
   | `XERXES_SLACK_BROKER_CLIENT_SECRET` | Slack app client secret |
   | `XERXES_SLACK_BROKER_ENROLLMENT_TOKEN` | Required enrollment credential: 32 cryptographically random bytes encoded as unpadded base64url (43 characters) |
   | `XERXES_SLACK_BROKER_PORT` | Optional loopback port; default `8796` |

4. Terminate TLS at a reverse proxy on the same host. Forward requests to `127.0.0.1:<port>` and preserve the incoming public `Host` header. The broker does not trust forwarded host, scheme or client-IP headers. Forward the `Authorization` header for `/v1/slack/start`. Disable access/query logging for these routes and never log authorization headers, cookies, response redirects, request bodies, or response bodies. Do not cache any broker response. Require HTTPS and configure proxy-level request timeouts and admission limits.

Use one broker process per origin: flow state and tokens live only in memory, expire after ten minutes from start, and are discarded on process restart. The process allows 256 live flows, 128 authenticated start attempts and 2,048 admitted requests per minute, and eight concurrent Slack exchanges. These shared limits bound resource use but cannot prevent an abusive client from exhausting capacity; configure admission limits at the proxy. A lost redeem response cannot be replayed; restart the connection flow. The broker provides no revocation endpoint; revoke unused grants in Slack if needed. Monitor status counts without recording flow identifiers or request/response content.

## Hub and Desktop protocol

Set `XERXES_HUB_SLACK_BROKER_ORIGIN` on the Hub to the trusted public HTTPS broker origin and `XERXES_HUB_SLACK_BROKER_ENROLLMENT_TOKEN` to the same enrollment token configured on the broker. Provision the token out of band only to trusted Hub hosts. It is not a Slack app secret, Desktop pairing code, or renderer setting. Any Hub holding this shared credential is trusted to initiate a grant; if leaked, rotate it on the broker and every enrolled Hub. Rotation requires restarting the broker, which discards existing flows.

1. The owner-authenticated Hub generates a random verifier (`32` random bytes encoded as unpadded base64url) and computes `base64url(SHA-256(verifier))` over the verifier **string**. It sends JSON `POST /v1/slack/start` with `{ "challenge": "…" }` and `Authorization: Bearer <enrollment-token>`. A missing or invalid credential returns `401` and creates no flow. Keep the credential in Hub and broker processes only; do not send it to the renderer, browser, or Slack. The response is `{ "id": "…", "browser_url": "https://<broker-host>/v1/slack/authorize?id=…", "pairing_code": "…" }`. Deliver the pairing code only to the initiating owner/session. Show it in Desktop separately from the browser link; never put it in URLs, telemetry, transcripts, or model context. Do not allow arbitrary broker origins from renderer/model input.
2. Desktop opens the returned URL in the owner's browser. The owner enters the code there before proceeding to Slack. A copied browser URL alone cannot authorize a different Slack account. Browser navigation uses a Secure, HttpOnly, SameSite=Lax cookie, and Slack's callback requires that cookie and one-time OAuth state. After the callback, the browser displays the workspace and user IDs and asks the owner to approve or reject that exact account before any token can be redeemed. Only enter a code on the known broker origin; never share it with another person.
3. The Hub polls `POST /v1/slack/redeem` with JSON `{ "id": "…", "verifier": "…" }`. It gets `{ "status": "pending" }`, `{ "status": "denied" }`, `{ "status": "expired" }`, or one `{ "status": "ready", "access_token": "…", "team_id": "…", "user_id": "…", "scope": "…" }`. Poll no faster than once per two seconds. Allow for `429` with `Retry-After`; honor the delay. `expired` also covers unknown, consumed, or verifier-mismatched flows. The Hub keeps the verifier and token off the renderer, checks the token identity, and presents the workspace and user IDs in Desktop. Confirm only if those IDs match the browser review; the Hub does not save the credential until this final owner confirmation.

For the Desktop release, set `XERXES_DESKTOP_SLACK_BROKER_ORIGIN` at build time to this same public HTTPS origin. This is a public trust pin, not a credential. With no pin the renderer must not open a managed broker URL. End users do not set this variable.

Offline checks: from `xerxes/`, run `bun test ./test/slackBroker.test.ts`. Tests inject a fake Slack fetch; they do not contact Slack or deploy a service.
