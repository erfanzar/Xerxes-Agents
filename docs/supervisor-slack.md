# Supervisor Slack: approved one-to-one DMs

The Hub Slack transport acts as the connected Slack **user**, not as a bot. Slack may grant user scopes across the workspace; Xerxes narrows their use to one-to-one DMs with explicitly approved peers. An OAuth grant by itself does not approve any peer or authorize automatic sending. Use this only with a workspace and account whose administrators and users have agreed to the integration.

## Sign in with Slack from the app (no Hub settings)

The Hub needs an HTTPS public origin (`XERXES_HUB_PUBLIC_ORIGINS` or the saved `public_origin`), because Slack sends your browser back to `https://<hub>/api/hub/v1/slack/oauth/callback` after you sign in. The callback only has to be reachable from your own browser, so a tailnet-only address works. Nothing else is set on the server.

**One-time setup (about a minute), as the Hub owner in Desktop → XERXES → Slack:**

1. Click **Create XERXES app in Slack**. Slack opens with a manifest already filled in. It requests only the six user scopes (`users:read`, `users:read.email`, `im:write`, `im:read`, `im:history`, `chat:write`), has no bot, no Socket Mode and no token rotation, and lists this Hub's callback as its only redirect URL. Pick your workspace, click **Next**, then **Create**. If Slack opens without the details, choose **From a manifest** and paste the manifest from **Copy manifest**.
2. On the new app's **Basic Information** page, copy the **Client ID** and **Client Secret** into the panel and click **Save**.

Before saving, the Hub asks Slack whether the Client ID and Secret are right, and refuses them with a plain message if not. They are stored in the Hub's state directory (`slack-app.json`, mode 0600) and are never returned to any client or written to the audit log.

**Connecting, from then on:** click **Sign in with Slack**, approve in the browser, then click **Confirm Slack account** in the panel. XERXES uses a user token, so it acts as you, not as a bot. Disconnecting and reconnecting is the same one click. **Use a different app** removes the saved app while no account is connected. Operator settings (below) take precedence over a saved app.

## Choose managed or self-hosted sign-in

An operator can select managed sign-in with [`XERXES_HUB_SLACK_BROKER_ORIGIN`](managed-slack-service.md), a trusted HTTPS broker origin. Managed sign-in requires a matching operator-provisioned enrollment secret (32 random bytes encoded as unpadded base64url): `XERXES_HUB_SLACK_BROKER_ENROLLMENT_TOKEN` on the Hub and `XERXES_SLACK_BROKER_ENROLLMENT_TOKEN` on the broker. The Hub sends it as a Bearer credential on both start and redeem, never to the browser or renderer. If the Hub is configured for managed sign-in but has no enrollment secret, managed sign-in is unavailable and does not fall back to self-hosted OAuth. Neither the owner nor the desktop enters Slack app credentials. With no broker origin configured, use the self-hosted app setup below.

For either mode, after browser authorization, verify the pending workspace and user IDs in Xerxes and explicitly confirm them. The Hub does not save the Slack credential or connect the account until you confirm that exact identity.

## Self-hosted Slack app setup

1. Create a dedicated Slack app for your workspace. In **OAuth & Permissions → Redirect URLs**, add `https://YOUR-HUB-HOST/api/hub/v1/slack/oauth/callback`, replacing the host with an explicitly configured Hub HTTPS origin. The same URL must be used during authorization and code exchange. Do not use the existing Slack bot/channel adapter's credentials.
2. Under **OAuth & Permissions → User Token Scopes**, add only the scopes below. Leave the bot scopes empty. Do not enable token rotation or install the app at the organization level; this transport accepts only workspace user grants without refresh tokens. If your workspace requires app approval, obtain that approval before connecting.

   | User scope | Use |
   | --- | --- |
   | `users:read`, `users:read.email` | Exact profile lookup by Slack user ID or email. No directory listing or fuzzy search. |
   | `im:write` | Open or resume a DM with one approved peer. |
   | `im:read` | Verify the conversation is a one-to-one DM with that peer. |
   | `im:history` | Fetch one bounded page from the verified DM. |
   | `chat:write` | Send text to that DM as the connected user. |

3. Set `XERXES_HUB_SLACK_CLIENT_ID`, `XERXES_HUB_SLACK_CLIENT_SECRET`, and `XERXES_HUB_SLACK_REDIRECT_URI` in the Hub service's environment. Supply the secret through the host's secret configuration, not a project file, client payload, or agent prompt. Connect as the Hub owner and authorize as the intended Slack user. Review and approve individual Slack user IDs before enabling DM activity. If the app previously had extra scopes, remove its old authorization and reconnect with the exact scope set.

## Host wiring

`HubSlackTransport` takes `stateDir`, `oauth: { clientId, clientSecret, redirectUri }`, and a host-owned `approvePeer(peer, operation)` callback. Use a Hub-owned local state directory, not an agent workspace. For a multi-owner host, isolate each owner's credential directory and require the authenticated owner to control its OAuth connection and approved contacts. The tuple `{ teamId, userId, peerId }` uses **Slack** IDs; the Hub owner is a separate identity that the host must enforce.

- `beginOAuth(binding, expectedIdentity?)` returns `{ authorizeUrl, expiresAt }`. The callback handler calls `completeOAuth(binding, { state, code?, error? }, signal?)`. Derive `binding` from the authenticated Hub owner/session, never from a callback query parameter. Validate callback inputs and reject duplicate query fields; avoid access logs, referrers, or analytics that capture the code or state. Use the same transport instance for start and callback. Pending state expires after 10 minutes, is consumed before exchange, and is lost on restart. `completeOAuth` returns only `{ teamId, userId }` after checking `auth.test` and storing the credential.
- `authTest` verifies the stored Slack identity. `lookupProfile` takes exactly one selector and returns only the user ID and display name. Keep both host-only. `openPeer` returns a peer-bound DM reference; `verifyDm`, `readDm`, and `sendDm` take that reference. Reads and sends revalidate the DM and check current approval again before accessing content. No callback means no DM access. Public, private and multi-party channels, Slack Connect DMs, search, files, and arbitrary Slack API calls are not supported. Treat Slack profile names and message text as untrusted data.
- `<stateDir>/slack-credentials` is owner-only (`0700`); token files are owner-only (`0600`). Tokens are not encrypted at rest: protect the Hub account, local disk and backups. `disconnect(identity)` deletes the local credential only. To revoke Slack-side access, remove the grant or app in Slack as permitted by your workspace. Local disconnect cannot undo a send already in flight.
- Reads are limited to 15 messages from one page. The transport does not retry requests or sends; respect `Retry-After` for polling. A send that times out or loses its acknowledgement might have reached Slack. Do not automatically retry it.

## Official Slack references

- [Installing with OAuth](https://docs.slack.dev/authentication/installing-with-oauth/) and [`oauth.v2.access`](https://docs.slack.dev/reference/methods/oauth.v2.access/)
- [`auth.test`](https://docs.slack.dev/reference/methods/auth.test/), [`users.info`](https://docs.slack.dev/reference/methods/users.info/), and [`users.lookupByEmail`](https://docs.slack.dev/reference/methods/users.lookupByEmail/)
- [`conversations.open`](https://docs.slack.dev/reference/methods/conversations.open/), [`conversations.info`](https://docs.slack.dev/reference/methods/conversations.info/), and [`conversations.history`](https://docs.slack.dev/reference/methods/conversations.history/)
- [`chat.postMessage`](https://docs.slack.dev/reference/methods/chat.postMessage/) and [Web API rate limits](https://docs.slack.dev/apis/web-api/rate-limits/)
