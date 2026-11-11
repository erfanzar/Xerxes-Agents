// Licensed under the Apache License, Version 2.0.

import { text, type RpcRecord } from './desktopRpc.js'

export interface SkillTrustFailure { readonly name: string; readonly error: string }
export interface SkillTrustBatch {
  readonly trusted: readonly string[]
  readonly failed: readonly SkillTrustFailure[]
}

/** Trust a fixed catalog snapshot sequentially, continuing after individual refusals. */
export async function trustAllSkills(
  names: readonly string[],
  trust: (name: string) => Promise<RpcRecord>,
): Promise<SkillTrustBatch> {
  const trusted: string[] = [], failed: SkillTrustFailure[] = []
  for (const name of names) {
    try {
      const result = await trust(name)
      if (result.ok === false) failed.push({ name, error: text(result.error) || text(result.output) || 'Trust was not accepted.' })
      else trusted.push(name)
    } catch (error) {
      failed.push({ name, error: error instanceof Error ? error.message : String(error) })
    }
  }
  return { trusted, failed }
}
