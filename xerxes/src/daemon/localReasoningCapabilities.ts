// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type {ReasoningLevelSet} from '../llms/reasoningLevels.js'
import {parseLocalProviderCapabilities,type LocalProviderCapabilities} from '../protocol/localProviderCapabilities.js'

export function localCapabilitySnapshot(model:string, levels:ReasoningLevelSet):LocalProviderCapabilities {
  return parseLocalProviderCapabilities({version:1,model,reasoning:{shape:levels.shape,
    efforts:levels.levels.map(level=>level.effort),canDisable:levels.shape!=='inherent'&&levels.canDisable!==false,
    provenance:levels.provenance??(levels.source==='provider'?'provider_reported':'provider_fallback'),
  }},model)!
}

export function localReasoningLevels(capabilities:LocalProviderCapabilities):ReasoningLevelSet {
  const r=capabilities.reasoning
  return {shape:r.shape,levels:r.efforts.map(effort=>({effort})),canDisable:r.canDisable,
    source:r.provenance==='provider_reported'?'provider':'fallback',provenance:r.provenance,defaultEffort:undefined}
}
export function localReasoningNote(levels:ReasoningLevelSet):string {
  const source=levels.provenance==='provider_reported'?'reported by the local provider'
    :levels.provenance==='bundled_catalog'?'from the local bundled model catalog':'from the local provider fallback table; not live-verified'
  const controls=levels.shape==='inherent'?'This model has no selectable reasoning controls. '
    :levels.shape==='toggle'?'This model exposes an on/off control. ':''
  return `${controls}Capability snapshot ${source}. Changes apply only to this task; local profile defaults stay local.`
}

/** Even an injected catalog that ignores cancellation cannot hold setup open. */
export async function boundedLocalCapabilities(model:string, read:(signal:AbortSignal)=>Promise<ReasoningLevelSet>,
  fallback:ReasoningLevelSet,timeoutMs=3000):Promise<LocalProviderCapabilities> {
  const abort=new AbortController()
  let timer:ReturnType<typeof setTimeout>|undefined
  try {
    const levels=await Promise.race([
      Promise.resolve().then(()=>read(abort.signal)),
      new Promise<ReasoningLevelSet>(resolve=>{timer=setTimeout(()=>{abort.abort();resolve(fallback)},timeoutMs);timer.unref?.()}),
    ])
    return localCapabilitySnapshot(model,levels)
  } catch {return localCapabilitySnapshot(model,fallback)}
  finally {clearTimeout(timer);abort.abort()}
}
