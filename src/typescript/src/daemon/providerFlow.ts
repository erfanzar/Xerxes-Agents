// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  CLAUDE_CODE_DEFAULT_MODEL,
  type ProviderProfile,
  type SaveProfileInput,
} from '../bridge/profiles.js'
import { PROVIDERS, type ProviderName } from '../llms/providerRegistry.js'
import type { DaemonQuestion } from './interactions.js'

export const PROVIDER_FLOW_ADD_LABEL = '+ Add new profile…'
export const PROVIDER_FLOW_CANCEL_LABEL = 'Cancel'
export const PROVIDER_FLOW_CUSTOM_MODEL_LABEL = '— Type a custom model id —'
export const PROVIDER_FLOW_EDIT_LABEL = '✎ Edit existing profile…'
export const PROVIDER_FLOW_REMOVE_LABEL = '✗ Remove existing profile…'

const EDIT_FIELDS = ['base_url', 'api_key', 'model', 'name', 'provider_type'] as const

type EditField = typeof EDIT_FIELDS[number]
type ProviderKind = ProviderName | 'auto'
type StoredProfile = ProviderProfile & { readonly active: boolean }

export interface ProviderProfileStore {
  active(): ProviderProfile | undefined
  delete(name: string): boolean
  list(): StoredProfile[]
  save(input: SaveProfileInput): ProviderProfile
  setActive(name: string): boolean
}

/** Host-provided model catalogue lookup. The flow never performs network I/O itself. */
export interface ProviderModelDiscoveryPort {
  discover(input: ProviderModelDiscoveryRequest): Promise<readonly string[]>
}

export interface ProviderModelDiscoveryRequest {
  readonly apiKey: string
  readonly baseUrl: string
  readonly provider: string
}

export interface ProviderProfileFlowOptions {
  readonly modelDiscovery?: ProviderModelDiscoveryPort
  readonly profileStore: ProviderProfileStore
}

export interface ProviderFlowNotice {
  readonly body: string
  readonly severity: 'error' | 'info' | 'warning'
}

export interface ProviderFlowPrompt {
  readonly question: DaemonQuestion
  readonly requestId: string
}

export interface ProviderFlowTransition {
  readonly finished: boolean
  readonly notice?: ProviderFlowNotice
  readonly prompt?: ProviderFlowPrompt
  /** The daemon should reload its active provider profile and refresh the client. */
  readonly reload?: boolean
}

interface AddDraft {
  readonly apiKey: string
  readonly baseUrl: string
  readonly defaultApiKey: string
  readonly defaultBaseUrl: string
  readonly defaultModel: string
  readonly name: string
  readonly provider: ProviderKind
}

interface MainState {
  readonly kind: 'main'
  readonly profileChoices: ReadonlyMap<string, string>
}

interface AddNameState {
  readonly kind: 'add_name'
}

interface AddProviderState {
  readonly kind: 'add_provider'
  readonly name: string
}

interface AddBaseUrlState {
  readonly draft: AddDraft
  readonly kind: 'add_base_url'
}

interface AddApiKeyState {
  readonly draft: AddDraft
  readonly kind: 'add_api_key'
}

interface AddModelState {
  readonly draft: AddDraft
  readonly kind: 'add_model'
}

interface AddCustomModelState {
  readonly draft: AddDraft
  readonly kind: 'add_custom_model'
}

interface EditProfileState {
  readonly kind: 'edit_profile'
}

interface EditFieldState {
  readonly kind: 'edit_field'
  readonly profile: StoredProfile
}

interface EditValueState {
  readonly field: EditField
  readonly kind: 'edit_value'
  readonly profile: StoredProfile
}

interface RemoveProfileState {
  readonly kind: 'remove_profile'
}

interface RemoveConfirmState {
  readonly kind: 'remove_confirm'
  readonly name: string
}

type FlowState =
  | MainState
  | AddNameState
  | AddProviderState
  | AddBaseUrlState
  | AddApiKeyState
  | AddModelState
  | AddCustomModelState
  | EditProfileState
  | EditFieldState
  | EditValueState
  | RemoveProfileState
  | RemoveConfirmState

/**
 * One connection-scoped `/provider` onboarding state machine.
 *
 * It deliberately has no transport or network dependency: the daemon owns
 * question delivery, runtime reloads, and any optional model-catalogue port.
 */
export class ProviderProfileFlow {
  private readonly modelDiscovery: ProviderModelDiscoveryPort | undefined
  private pending: ProviderFlowPrompt | undefined
  private readonly profileStore: ProviderProfileStore
  private state: FlowState | undefined

  constructor(options: ProviderProfileFlowOptions) {
    this.profileStore = options.profileStore
    this.modelDiscovery = options.modelDiscovery
  }

  get activeRequestId(): string | undefined {
    return this.pending?.requestId
  }

  /** Opens the top-level picker, replacing any unfinished local flow state. */
  async start(): Promise<ProviderFlowTransition> {
    this.pending = undefined
    this.state = undefined
    return this.askMain()
  }

  /** Stops an unfinished flow without persisting its draft. */
  cancel(): ProviderFlowTransition | undefined {
    if (!this.state) {
      return undefined
    }
    return this.complete('Cancelled.')
  }

  /**
   * Advance the current single-question step.
   *
   * Returns `undefined` for an unknown request or malformed response so the
   * daemon can leave the owned question pending, like DaemonInteractionBoard.
   */
  async answer(requestId: string, answers: Readonly<Record<string, string>>): Promise<ProviderFlowTransition | undefined> {
    const pending = this.pending
    if (!pending || pending.requestId !== requestId) {
      return undefined
    }
    const answer = answerFor(pending.question, answers)
    if (answer === undefined) {
      return undefined
    }
    if (pending.question.allowFreeform === false && !(pending.question.options ?? []).includes(answer)) {
      return undefined
    }

    this.pending = undefined
    if (isCancelled(answer)) {
      return this.complete('Cancelled.')
    }
    const state = this.state
    if (!state) {
      return this.complete('Provider setup was reset.', 'warning')
    }

    switch (state.kind) {
      case 'main':
        return this.advanceMain(state, answer)
      case 'add_name':
        return this.advanceAddName(answer)
      case 'add_provider':
        return this.advanceAddProvider(state, answer)
      case 'add_base_url':
        return this.advanceAddBaseUrl(state, answer)
      case 'add_api_key':
        return this.advanceAddApiKey(state, answer)
      case 'add_model':
        return this.advanceAddModel(state, answer)
      case 'add_custom_model':
        return this.finalizeAdd(state.draft, answer.trim() || state.draft.defaultModel)
      case 'edit_profile':
        return this.advanceEditProfile(answer)
      case 'edit_field':
        return this.advanceEditField(state, answer)
      case 'edit_value':
        return this.advanceEditValue(state, answer)
      case 'remove_profile':
        return this.advanceRemoveProfile(answer)
      case 'remove_confirm':
        return this.advanceRemoveConfirm(state, answer)
    }
  }

  private advanceMain(state: MainState, answer: string): ProviderFlowTransition {
    if (answer === PROVIDER_FLOW_ADD_LABEL) {
      return this.askAddName()
    }
    if (answer === PROVIDER_FLOW_EDIT_LABEL) {
      return this.askEditProfile()
    }
    if (answer === PROVIDER_FLOW_REMOVE_LABEL) {
      return this.askRemoveProfile()
    }
    const name = state.profileChoices.get(answer)
    if (!name) {
      return this.complete('No provider profile selected.', 'warning')
    }
    if (!this.profileStore.setActive(name)) {
      return this.complete(`No provider profile named \`${name}\`.`, 'error')
    }
    const active = this.profileStore.active()
    return this.complete(
      `Switched to provider profile \`${name}\` (model: \`${active?.model || '(no model)'}\`).`,
      'info',
      true,
    )
  }

  private advanceAddName(answer: string): ProviderFlowTransition {
    const name = answer.trim()
    if (!name) {
      return this.complete('Add cancelled — profile name is required.', 'warning')
    }
    this.state = { kind: 'add_provider', name }
    return this.ask({
      allowFreeform: true,
      options: providerTypeOptions(),
      question: 'Inference provider type:',
      questionId: 'provider_type',
    })
  }

  private async advanceAddProvider(state: AddProviderState, answer: string): Promise<ProviderFlowTransition> {
    const provider = canonicalProviderType(answer)
    if (!provider) {
      return this.complete(`Unknown provider type \`${answer.trim()}\`.`, 'warning')
    }
    const draft = defaultDraft(state.name, provider)
    if (provider === 'claude-code') {
      return this.askModel({ ...draft, baseUrl: 'claude-code://local', apiKey: '' })
    }
    this.state = { kind: 'add_base_url', draft }
    const defaultUrl = draft.defaultBaseUrl
    return this.ask({
      allowFreeform: true,
      options: [],
      question: defaultUrl ? `Base URL (press Enter for \`${defaultUrl}\`):` : 'Base URL:',
      questionId: 'base_url',
      ...(defaultUrl ? { placeholder: defaultUrl } : {}),
    })
  }

  private advanceAddBaseUrl(state: AddBaseUrlState, answer: string): ProviderFlowTransition {
    const baseUrl = answer.trim() || state.draft.defaultBaseUrl
    if (!baseUrl) {
      return this.complete(
        `Add cancelled — base_url is required for \`${state.draft.provider}\` because it has no registry default.`,
        'warning',
      )
    }
    const draft = { ...state.draft, baseUrl }
    this.state = { kind: 'add_api_key', draft }
    return this.ask({
      allowFreeform: true,
      options: [],
      question: 'API key (blank uses the provider default or environment when available):',
      questionId: 'api_key',
    })
  }

  private async advanceAddApiKey(state: AddApiKeyState, answer: string): Promise<ProviderFlowTransition> {
    return this.askModel({ ...state.draft, apiKey: answer.trim() || state.draft.defaultApiKey })
  }

  private advanceAddModel(state: AddModelState, answer: string): ProviderFlowTransition {
    const model = answer.trim()
    if (model === PROVIDER_FLOW_CUSTOM_MODEL_LABEL) {
      this.state = { kind: 'add_custom_model', draft: state.draft }
      return this.ask({
        allowFreeform: true,
        options: [],
        question: state.draft.defaultModel
          ? `Custom model id (press Enter for \`${state.draft.defaultModel}\`):`
          : 'Custom model id:',
        questionId: 'model',
        ...(state.draft.defaultModel ? { placeholder: state.draft.defaultModel } : {}),
      })
    }
    return this.finalizeAdd(state.draft, model || state.draft.defaultModel)
  }

  private advanceEditProfile(answer: string): ProviderFlowTransition {
    const profile = this.findProfile(answer)
    if (!profile) {
      return this.complete(`No provider profile named \`${answer}\`.`, 'warning')
    }
    this.state = { kind: 'edit_field', profile }
    return this.ask({
      allowFreeform: false,
      options: EDIT_FIELDS,
      question: 'Which field should be updated?',
      questionId: 'field',
    })
  }

  private advanceEditField(state: EditFieldState, answer: string): ProviderFlowTransition {
    if (!isEditField(answer)) {
      return this.complete(`Unknown profile field \`${answer}\`.`, 'warning')
    }
    this.state = { kind: 'edit_value', field: answer, profile: state.profile }
    return this.ask({
      allowFreeform: true,
      options: [],
      question: answer === 'api_key' ? 'New API key:' : `New value for \`${answer}\`:`,
      questionId: 'value',
    })
  }

  private advanceEditValue(state: EditValueState, answer: string): ProviderFlowTransition {
    const current = this.findProfile(state.profile.name)
    if (!current) {
      return this.complete(`No provider profile named \`${state.profile.name}\`.`, 'warning')
    }
    const value = answer.trim()
    if (state.field === 'name' && !value) {
      return this.complete('Edit cancelled — new profile name is required.', 'warning')
    }

    const provider = state.field === 'provider_type' ? canonicalProviderType(value) : current.provider
    if (!provider) {
      return this.complete(`Unknown provider type \`${value}\`.`, 'warning')
    }
    const name = state.field === 'name' ? value : current.name
    const input: SaveProfileInput = {
      apiKey: state.field === 'api_key' ? value : current.api_key,
      baseUrl: state.field === 'base_url' ? value : current.base_url,
      model: state.field === 'model' ? value : current.model,
      name,
      setActive: current.active,
      ...(state.field === 'provider_type' && provider === 'auto' ? {} : { provider }),
    }
    try {
      this.profileStore.save(input)
      if (name !== current.name) {
        this.profileStore.delete(current.name)
      }
    } catch {
      return this.complete('Failed to update the provider profile.', 'error')
    }

    const shown = state.field === 'api_key' ? '***redacted***' : value
    return this.complete(`Updated \`${name}\`: \`${state.field}\` = \`${shown}\`.`, 'info', true)
  }

  private advanceRemoveProfile(answer: string): ProviderFlowTransition {
    const profile = this.findProfile(answer)
    if (!profile) {
      return this.complete(`No provider profile named \`${answer}\`.`, 'warning')
    }
    this.state = { kind: 'remove_confirm', name: profile.name }
    return this.ask({
      allowFreeform: true,
      options: ['yes', 'no'],
      question: `Type \`yes\` to remove \`${profile.name}\`:`,
      questionId: 'confirm',
    })
  }

  private advanceRemoveConfirm(state: RemoveConfirmState, answer: string): ProviderFlowTransition {
    if (!['yes', 'y'].includes(answer.trim().toLowerCase())) {
      return this.complete(`Remove cancelled — \`${state.name}\` was not deleted.`, 'info')
    }
    if (!this.profileStore.delete(state.name)) {
      return this.complete(`Failed to remove \`${state.name}\`.`, 'error')
    }
    return this.complete(`Removed provider profile \`${state.name}\`.`, 'info', true)
  }

  private async askMain(): Promise<ProviderFlowTransition> {
    const choices = new Map<string, string>()
    for (const profile of this.profileStore.list()) {
      const label = `${profile.name}  (${profile.model || '?'} @ ${profile.base_url})${profile.active ? '  ← active' : ''}`
      choices.set(label, profile.name)
    }
    this.state = { kind: 'main', profileChoices: choices }
    const options = [...choices.keys(), PROVIDER_FLOW_ADD_LABEL]
    if (choices.size) {
      options.push(PROVIDER_FLOW_EDIT_LABEL, PROVIDER_FLOW_REMOVE_LABEL)
    }
    return this.ask({
      allowFreeform: false,
      options,
      question: 'Provider profiles — pick a profile to switch, or choose an action:',
      questionId: 'action',
    })
  }

  private askAddName(): ProviderFlowTransition {
    this.state = { kind: 'add_name' }
    return this.ask({
      allowFreeform: true,
      options: [],
      question: 'New profile name (for example, \`openai-prod\`):',
      questionId: 'name',
    })
  }

  private askEditProfile(): ProviderFlowTransition {
    const names = this.profileStore.list().map(profile => profile.name)
    if (!names.length) {
      return this.complete('No provider profiles are available to edit.', 'warning')
    }
    this.state = { kind: 'edit_profile' }
    return this.ask({
      allowFreeform: false,
      options: names,
      question: 'Which profile should be edited?',
      questionId: 'profile',
    })
  }

  private askRemoveProfile(): ProviderFlowTransition {
    const names = this.profileStore.list().map(profile => profile.name)
    if (!names.length) {
      return this.complete('No provider profiles are available to remove.', 'warning')
    }
    this.state = { kind: 'remove_profile' }
    return this.ask({
      allowFreeform: false,
      options: names,
      question: 'Which profile should be removed?',
      questionId: 'profile',
    })
  }

  private async askModel(draft: AddDraft): Promise<ProviderFlowTransition> {
    const staticModels = draft.provider === 'auto' ? [] : PROVIDERS[draft.provider].models
    let discovered: readonly string[] = []
    let discoveryUnavailable = false
    if (this.modelDiscovery && draft.provider !== 'claude-code') {
      try {
        discovered = await this.modelDiscovery.discover({
          apiKey: draft.apiKey,
          baseUrl: draft.baseUrl,
          provider: draft.provider,
        })
      } catch {
        // Do not surface a port error: provider SDKs sometimes include credentials in failures.
        discoveryUnavailable = true
      }
    }
    const models = uniqueStrings([
      draft.defaultModel,
      ...staticModels,
      ...discovered,
    ])
    const normalizedDraft = {
      ...draft,
      defaultModel: draft.defaultModel || models[0] || '',
    }
    this.state = { kind: 'add_model', draft: normalizedDraft }
    const question = discoveryUnavailable
      ? 'Pick a model (catalogue lookup was unavailable; you can type one manually):'
      : models.length
        ? 'Pick a model (or type a custom model id):'
        : 'Pick a model (type a model id):'
    return this.ask({
      allowFreeform: true,
      options: models.length ? [...models, PROVIDER_FLOW_CUSTOM_MODEL_LABEL] : [PROVIDER_FLOW_CUSTOM_MODEL_LABEL],
      question,
      questionId: 'model',
    })
  }

  private finalizeAdd(draft: AddDraft, selectedModel: string): ProviderFlowTransition {
    let baseUrl = draft.baseUrl
    let apiKey = draft.apiKey
    let model = selectedModel.trim()
    if (draft.provider === 'claude-code') {
      baseUrl = baseUrl || 'claude-code://local'
      apiKey = ''
      if (model && !model.startsWith('claude-code/')) {
        if (model.includes('/')) {
          return this.complete(
            'Add cancelled — Claude Code model ids must be bare aliases or start with \`claude-code/\`.',
            'warning',
          )
        }
        model = `claude-code/${model}`
      }
    }
    if (!model) {
      return this.complete(`Add cancelled — \`${draft.provider}\` has no default model.`, 'warning')
    }
    try {
      const profile = this.profileStore.save({
        apiKey,
        baseUrl,
        model,
        name: draft.name,
        ...(draft.provider === 'auto' ? {} : { provider: draft.provider }),
      })
      return this.complete(
        `Added profile \`${profile.name}\` (type \`${profile.provider}\`, model \`${profile.model}\` @ \`${profile.base_url}\`) and switched to it.`,
        'info',
        true,
      )
    } catch {
      return this.complete('Failed to save the provider profile.', 'error')
    }
  }

  private ask(question: DaemonQuestion): ProviderFlowTransition {
    const normalizedQuestion: DaemonQuestion = {
      ...question,
      allowFreeform: question.allowFreeform ?? true,
      options: uniqueStrings([...(question.options ?? []), PROVIDER_FLOW_CANCEL_LABEL]),
    }
    const prompt: ProviderFlowPrompt = {
      question: normalizedQuestion,
      requestId: crypto.randomUUID().replaceAll('-', ''),
    }
    this.pending = prompt
    return { finished: false, prompt }
  }

  private complete(
    body: string,
    severity: ProviderFlowNotice['severity'] = 'info',
    reload = false,
  ): ProviderFlowTransition {
    this.pending = undefined
    this.state = undefined
    return {
      finished: true,
      notice: { body, severity },
      ...(reload ? { reload: true } : {}),
    }
  }

  private findProfile(name: string): StoredProfile | undefined {
    return this.profileStore.list().find(profile => profile.name === name)
  }
}

/** Normalize the Python-era `claude_code` alias and the empty/auto selection. */
export function canonicalProviderType(value: string): ProviderKind | undefined {
  const clean = value.trim().toLowerCase().replace('claude_code', 'claude-code')
  if (!clean || clean === 'auto') {
    return 'auto'
  }
  return Object.hasOwn(PROVIDERS, clean) ? clean as ProviderName : undefined
}

export function providerTypeOptions(): readonly ProviderKind[] {
  return ['auto', ...(Object.keys(PROVIDERS) as ProviderName[])]
}

function answerFor(question: DaemonQuestion, answers: Readonly<Record<string, string>>): string | undefined {
  const id = question.questionId ?? 'answer'
  return answers[id] ?? Object.values(answers).find(answer => typeof answer === 'string')
}

function defaultDraft(name: string, provider: ProviderKind): AddDraft {
  const config = provider === 'auto' ? undefined : PROVIDERS[provider]
  return {
    apiKey: '',
    baseUrl: '',
    defaultApiKey: config?.defaultApiKey ?? '',
    defaultBaseUrl: config?.baseUrl ?? '',
    defaultModel: provider === 'claude-code' ? CLAUDE_CODE_DEFAULT_MODEL : config?.models[0] ?? '',
    name,
    provider,
  }
}

function isCancelled(value: string): boolean {
  return value.trim().toLowerCase() === PROVIDER_FLOW_CANCEL_LABEL.toLowerCase()
}

function isEditField(value: string): value is EditField {
  return (EDIT_FIELDS as readonly string[]).includes(value)
}

function uniqueStrings(values: readonly string[]): string[] {
  const seen = new Set<string>()
  for (const value of values) {
    const normalized = value.trim()
    if (normalized) {
      seen.add(normalized)
    }
  }
  return [...seen]
}
