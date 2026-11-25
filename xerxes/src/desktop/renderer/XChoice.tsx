// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useRef, useState, type KeyboardEvent, type ReactElement } from 'react'

import { Icon } from './Icon.js'

/**
 * The XERXES controls' own pickers, in place of native dropdowns: chips for a
 * handful of short choices (review interval, expiry, effort) and a list with a
 * filter for longer ones (tasks, machines, profiles, models). Both are radio
 * groups, so screen readers and the keyboard treat them as one choice.
 */

export interface ChoiceOption<T extends string> {
  readonly value: T
  readonly label: string
  /** Shown beside the label in a list; the tooltip on a chip. */
  readonly hint?: string
  readonly disabled?: boolean
}

function moveRadio<T extends string>(
  event: KeyboardEvent<HTMLElement>,
  container: HTMLElement | null,
  options: readonly ChoiceOption<T>[],
  currentValue: T,
  groupDisabled: boolean,
  onChange: (value: T) => void,
): void {
  if (groupDisabled) return
  const key = event.key
  if (key !== 'ArrowRight' && key !== 'ArrowDown' && key !== 'ArrowLeft' && key !== 'ArrowUp' && key !== 'Home' && key !== 'End') return
  const enabledIndices: number[] = []
  options.forEach((opt, idx) => { if (opt.disabled !== true) enabledIndices.push(idx) })
  if (enabledIndices.length === 0) return
  event.preventDefault()
  const curIdx = options.findIndex(opt => opt.value === currentValue)
  const curPos = enabledIndices.indexOf(curIdx)
  let nextPos = 0
  if (key === 'Home') nextPos = 0
  else if (key === 'End') nextPos = enabledIndices.length - 1
  else if (key === 'ArrowRight' || key === 'ArrowDown') nextPos = curPos < 0 ? 0 : (curPos + 1) % enabledIndices.length
  else nextPos = curPos <= 0 ? enabledIndices.length - 1 : curPos - 1
  const targetIdx = enabledIndices[nextPos]!
  const target = options[targetIdx]!
  if (target.value !== currentValue) onChange(target.value)
  const buttons = container?.querySelectorAll<HTMLButtonElement>('[role="radio"]')
  buttons?.[targetIdx]?.focus()
}

export function ChipChoice<T extends string>({ label, ariaLabelledBy, value, options, disabled = false, onChange }: {
  readonly label: string
  readonly ariaLabelledBy?: string
  readonly value: T
  readonly options: readonly ChoiceOption<T>[]
  readonly disabled?: boolean
  readonly onChange: (value: T) => void
}): ReactElement {
  const groupRef = useRef<HTMLDivElement>(null)
  const checkedIdx = options.findIndex(opt => opt.value === value && opt.disabled !== true)
  const tabbableIdx = checkedIdx >= 0 ? checkedIdx : options.findIndex(opt => opt.disabled !== true)
  return (
    <div
      ref={groupRef}
      className="xchoice"
      role="radiogroup"
      {...(ariaLabelledBy ? { 'aria-labelledby': ariaLabelledBy } : { 'aria-label': label })}
      onKeyDown={event => moveRadio(event, groupRef.current, options, value, disabled, onChange)}
    >
      {options.map((option, idx) => {
        const on = option.value === value
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={on}
            tabIndex={disabled || option.disabled === true ? -1 : idx === tabbableIdx ? 0 : -1}
            className={`xchip${on ? ' is-on' : ''}`}
            title={option.hint}
            disabled={disabled || option.disabled === true}
            onClick={() => { if (!on) onChange(option.value) }}
          >{option.label}</button>
        )
      })}
    </div>
  )
}

/** Lists longer than this get a filter box. */
export const LIST_FILTER_AT = 8

export function ListChoice<T extends string>({ label, ariaLabelledBy, value, options, disabled = false, empty, onChange }: {
  readonly label: string
  readonly ariaLabelledBy?: string
  readonly value: T
  readonly options: readonly ChoiceOption<T>[]
  readonly disabled?: boolean
  /** Shown while there is nothing to choose. */
  readonly empty?: string
  readonly onChange: (value: T) => void
}): ReactElement {
  const [filter, setFilter] = useState('')
  const groupRef = useRef<HTMLDivElement>(null)
  const needle = filter.trim().toLowerCase()
  const shown = needle === '' ? options : options.filter(option => `${option.label} ${option.hint ?? ''}`.toLowerCase().includes(needle))
  const checkedIdx = shown.findIndex(opt => opt.value === value && opt.disabled !== true)
  const tabbableIdx = checkedIdx >= 0 ? checkedIdx : shown.findIndex(opt => opt.disabled !== true)
  return (
    <div className="xlist">
      {options.length > LIST_FILTER_AT ? (
        <input type="search" className="xlist__filter" aria-label={`Filter ${label}`} placeholder="Filter…" value={filter} onChange={event => setFilter(event.target.value)} />
      ) : null}
      <div
        ref={groupRef}
        className="xlist__rows"
        role="radiogroup"
        {...(ariaLabelledBy ? { 'aria-labelledby': ariaLabelledBy } : { 'aria-label': label })}
        onKeyDown={event => moveRadio(event, groupRef.current, shown, value, disabled, onChange)}
      >
        {shown.length === 0 ? <p className="xlist__empty">{options.length === 0 ? empty ?? 'Nothing to choose yet.' : 'No match.'}</p> : shown.map((option, idx) => {
          const on = option.value === value
          return (
            <button
              key={option.value}
              type="button"
              role="radio"
              aria-checked={on}
              className={`xlist__row${on ? ' is-on' : ''}`}
              tabIndex={disabled || option.disabled === true ? -1 : idx === tabbableIdx ? 0 : -1}
              disabled={disabled || option.disabled === true}
              onClick={() => { if (!on) onChange(option.value) }}
            >
              <span className="xlist__check" aria-hidden="true">{on ? <Icon name="check" size={12} /> : null}</span>
              <span className="xlist__label">{option.label}</span>
              {option.hint === undefined || option.hint === '' ? null : <span className="xlist__hint">{option.hint}</span>}
            </button>
          )
        })}
      </div>
    </div>
  )
}
