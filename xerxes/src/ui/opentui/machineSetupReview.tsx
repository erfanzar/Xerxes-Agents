// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
import type { ScrollBoxRenderable } from '@opentui/core'
import { useKeyboard, useTerminalDimensions } from '@opentui/react'
import { useRef, useState } from 'react'
import type { RemoteMachine } from '../lib/machineHandoff.js'
import type { Theme } from '../theme.js'
import { ModalShell } from './pickerChrome.js'

// Execution boundaries, not a remote health check. Keep these aligned with
// remoteBootstrap, ProfileStore, configured MCP and bundled skill discovery.
const integrations = [
  { name: 'Overview', details: 'The TUI renders locally. Connection installs or updates a managed runtime under remote ~/.xerxes/remote-runtime. Code, tools and sessions run on the selected remote machine. Existing remote configuration stays in use. Remote integration readiness has not been inspected.', setup: 'Connect using remote setup, then inspect each integration there. The next step inspects the task and lets you review a scoped local provider grant before opening it.' },
  { name: 'Providers and API credentials', details: 'Keep remote saved profiles, authentication or environment credentials, or share the supported local setup in one review. With local access, provider requests run on this workstation; code and tools run remotely. Keys are not copied. Local grants stay in memory and expire; the remote task saves only its provider requirement.', setup: 'The next review lists the configured provider/model pairs, credential source, limits and destination. One approval covers all listed providers. After connecting, /provider can explicitly switch the task to a remote profile. /providers lists remote profiles.' },
  { name: 'Subscription authentication', details: 'With remote setup, login and token refresh run remotely. With an authorized supported local profile, they run on this workstation using its existing login. A login is never copied to the remote host. Expiration or revocation fails the selected route without choosing another provider.', setup: 'The next review lists supported local profiles and any extra output-length consent. Renew an expired login on the machine that owns it, then review access again. Unsupported profiles need remote setup.' },
  { name: 'Models and reasoning', details: 'The next review shares supported local profiles and their configured models together. Remote sessions retain their saved model and reasoning choices. Switching among approved local providers does not ask for another approval.', setup: 'For another local model, configure its local profile before connecting and review a new grant. In the remote task, /model lists approved local choices; /provider selects remote credentials explicitly. /reasoning controls supported effort levels. Selection alone does not prove connectivity.' },
  { name: 'Search', details: 'Execution: remote network. Public text instant answers need no local API key, but require remote public-network access. Full search and image/news/maps/video capabilities need a separately configured provider.', setup: 'Inspect remote tools with /features. Unsupported search modes must report their limitation; a listed tool is not proof that its service works.' },
  { name: 'MCP connections', details: 'Execution: remote daemon. Configuration: remote user mcp.json and trusted remote workspace .mcp.json. Credentials and executable dependencies are remote. Local stdio servers and local files are not automatically reachable.', setup: 'Use /config mcp to configure servers and /mcp to inspect health. Install required executables or authenticate the remote connection there.' },
  { name: 'Skills and assets', details: 'Execution: remote workspace. Sources: bundled runtime skills plus admitted remote user/project skills. Local custom skills and their assets are not copied.', setup: 'Use /skills to inspect discovery. Install selected custom skills and required assets on the remote machine, then /reload-skills. Review executable dependencies separately.' },
  { name: 'Agent definitions', details: 'Execution: remote daemon and workspace. Sources: remote agent definitions, compositions and bundled presets. Local custom specialists are not imported.', setup: 'Use /custom-agents and /config agents to inspect remote definitions. Install the selected definitions and referenced assets in the remote setup.' },
  { name: 'Plugins, hooks and LSP', details: 'Execution: remote processes. Configuration, workspace trust and dependencies belong to the remote machine. Local executables do not become remote integrations.', setup: 'Inspect /plugins and /hooks; use /config lsp for language servers. Install dependencies remotely and review workspace trust before enabling integrations.' },
  { name: 'Browser', details: 'Execution: the explicitly connected browser endpoint. No local browser is shared automatically, and Xerxes does not launch a browser.', setup: 'Start a Chromium-compatible browser separately and connect its authorized CDP endpoint using /browser connect <endpoint> in the remote session.' },
  { name: 'Channels and webhooks', details: 'Execution and delivery: remote daemon. Configuration and credentials: remote account setup. Local channel accounts and webhook secrets are not copied.', setup: 'Use /channels to inspect the remote gateways. Configure accounts and remote inbound connectivity explicitly before enabling delivery.' },
  { name: 'SSH and persistence', details: 'SSH authentication runs locally using your SSH configuration. SSH private keys stay local; agent and X11 forwarding are disabled. Remote sessions persist on the remote host; returning here preserves the local session.', setup: 'Verify the destination host key before first connection. Exit the remote TUI to return here. A dropped tunnel retries the same destination; it does not choose another host or provider.' },
] as const

export function MachineSetupReview({ machine, t, onBack, onConnect, busy, error, notice }: {
  machine: RemoteMachine; t: Theme; onBack: () => void; onConnect: () => void; busy: boolean; error: string; notice: string
}) {
  const { width, height } = useTerminalDimensions()
  const [index, setIndex] = useState(0)
  const scroll = useRef<ScrollBoxRenderable | null>(null)
  useKeyboard(key => {
    if (!['escape', 'return', 'enter', 'tab', 'left', 'right', 'up', 'down', 'pageup', 'pagedown', 'home', 'end'].includes(key.name)) return
    key.preventDefault(); key.stopPropagation()
    if (key.name === 'escape') { onBack(); return }
    if (busy) return
    if (key.name === 'return' || key.name === 'enter') { onConnect(); return }
    if (key.name === 'tab' || key.name === 'left' || key.name === 'right') {
      setIndex(old => (old + (key.shift || key.name === 'left' ? integrations.length - 1 : 1)) % integrations.length)
      scroll.current?.scrollTo(0); return
    }
    if (key.name === 'home') scroll.current?.scrollTo(0)
    else if (key.name === 'end') scroll.current?.scrollTo(1_000_000)
    else scroll.current?.scrollBy((key.name === 'up' || key.name === 'pageup' ? -1 : 1) * (key.name.startsWith('page') ? Math.max(1, height - 10) : 1))
  })
  const selected = integrations[index]!
  return <ModalShell t={t} title="Remote setup" height={height} width={width} panelWidth={Math.min(88, width)} panelHeight={Math.min(28, height)}>
    <box flexDirection="column" flexGrow={1} minHeight={0} paddingX={2}>
      <text flexShrink={0} fg={t.color.accent}>{`${index + 1}/${integrations.length} · ${selected.name}`}</text>
      <scrollbox key={index} ref={scroll} style={{ flexGrow: 1, minHeight: 0 }}>
        <text fg={t.color.text} wrapMode="word">{`Host: ${machine.target}\nWorkspace: ${machine.workspacePath}\n\n${selected.details}\n\n${selected.setup}`}</text>
        {error ? <text fg={t.color.error} wrapMode="word">{error}</text> : null}
        {notice ? <text fg={t.color.text} wrapMode="word">{notice}</text> : null}
      </scrollbox>
      <text flexShrink={0} fg={t.color.muted}>{busy ? 'Connecting… · Esc cancel' : 'Tab/←→ integration · ↑↓ scroll'}</text>
      <text flexShrink={0} fg={t.color.accent}>{busy ? '' : 'Enter prepare task · Esc back'}</text>
    </box>
  </ModalShell>
}
