// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Hardware probe for choosing a local model. Every probe goes through an
 * injected `run` port so tests use fakes; the real port spawns argv directly
 * (no shell) with a hard timeout. A missing probe binary is an expected,
 * non-fatal condition (it means "no such hardware"), never an exception.
 */
import { totalmem } from 'node:os'

const MIB = 1024 * 1024
const PROBE_TIMEOUT_MS = 5000

export interface CommandResult {
  readonly exitCode: number
  readonly stdout: string
}

export interface HardwarePorts {
  run(argv: string[]): Promise<CommandResult>
  totalMemBytes(): number
  readonly platform: NodeJS.Platform
  readonly arch: string
}

export type GpuVendor = 'apple' | 'nvidia' | 'none'

export interface GpuInfo {
  readonly vendor: GpuVendor
  readonly name?: string
  readonly vramBytes?: number
}

export interface HardwareProfile {
  readonly platform: NodeJS.Platform
  readonly arch: string
  readonly totalMemBytes: number
  readonly appleSilicon: boolean
  readonly gpu: GpuInfo
}

/** Run a probe; a missing binary, non-zero exit or spawn failure all mean "not available". */
async function tryRun(ports: HardwarePorts, argv: string[]): Promise<string | undefined> {
  try {
    const result = await ports.run(argv)
    if (result.exitCode !== 0) return undefined
    const text = result.stdout.trim()
    return text === '' ? undefined : text
  } catch {
    return undefined
  }
}

/** Parse `name, memory.total` CSV lines (MiB); the GPU with the most memory wins. */
export function parseNvidiaSmi(stdout: string): { name: string; vramBytes: number } | undefined {
  let best: { name: string; vramBytes: number } | undefined
  for (const line of stdout.split('\n')) {
    const comma = line.lastIndexOf(',')
    if (comma <= 0) continue
    const name = line.slice(0, comma).trim()
    const mib = Number.parseInt(line.slice(comma + 1).trim(), 10)
    if (name === '' || !Number.isFinite(mib) || mib <= 0) continue
    const vramBytes = mib * MIB
    if (best === undefined || vramBytes > best.vramBytes) best = { name, vramBytes }
  }
  return best
}

export async function probeHardware(ports: HardwarePorts): Promise<HardwareProfile> {
  const { platform, arch } = ports
  let totalMemBytes = ports.totalMemBytes()
  let appleSilicon = false
  let brand: string | undefined

  if (platform === 'darwin') {
    const memText = await tryRun(ports, ['sysctl', '-n', 'hw.memsize'])
    if (memText !== undefined && /^\d+$/.test(memText)) {
      const parsed = Number(memText)
      if (Number.isSafeInteger(parsed) && parsed > 0) totalMemBytes = parsed
    }
    brand = await tryRun(ports, ['sysctl', '-n', 'machdep.cpu.brand_string'])
    appleSilicon = brand !== undefined ? /^Apple M/i.test(brand) : arch === 'arm64'
  }

  if (appleSilicon) {
    return {
      platform,
      arch,
      totalMemBytes,
      appleSilicon,
      gpu: { vendor: 'apple', ...(brand === undefined ? {} : { name: brand }) },
    }
  }

  if (platform !== 'darwin') {
    const smi = await tryRun(ports, [
      'nvidia-smi',
      '--query-gpu=name,memory.total',
      '--format=csv,noheader,nounits',
    ])
    const gpu = smi === undefined ? undefined : parseNvidiaSmi(smi)
    if (gpu !== undefined) {
      return {
        platform,
        arch,
        totalMemBytes,
        appleSilicon,
        gpu: { vendor: 'nvidia', name: gpu.name, vramBytes: gpu.vramBytes },
      }
    }
  }

  return { platform, arch, totalMemBytes, appleSilicon, gpu: { vendor: 'none' } }
}

async function runWithTimeout(argv: string[]): Promise<CommandResult> {
  let child: Bun.Subprocess<'ignore', 'pipe', 'ignore'>
  try {
    child = Bun.spawn(argv, { stdin: 'ignore', stdout: 'pipe', stderr: 'ignore' })
  } catch {
    return { exitCode: 127, stdout: '' }
  }
  const timer = setTimeout(() => child.kill(), PROBE_TIMEOUT_MS)
  try {
    const [stdout, exitCode] = await Promise.all([new Response(child.stdout).text(), child.exited])
    return { exitCode, stdout }
  } finally {
    clearTimeout(timer)
  }
}

/** Real hardware ports: argv-only spawn (no shell), 5 s timeout per probe. */
export function systemHardwarePorts(): HardwarePorts {
  return {
    run: runWithTimeout,
    totalMemBytes: () => totalmem(),
    platform: process.platform,
    arch: process.arch,
  }
}
