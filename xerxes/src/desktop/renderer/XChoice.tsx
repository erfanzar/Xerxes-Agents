// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useState, type ReactElement } from 'react'

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

export function ChipChoice<T extends string>({ label, value, options, disabled = false, onChange }: {
  readonly label: string
  readonly value: T
  readonly options: readonly ChoiceOption<T>[]
  readonly disabled?: boolean
  readonly onChange: (value: T) => void
}): ReactElement {
  return (
    <div className="xchoice" role="radiogroup" aria-label={label}>
      {options.map(option => {
        const on = option.value === value
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={on}
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

export function ListChoice<T extends string>({ label, value, options, disabled = false, empty, onChange }: {
  readonly label: string
  readonly value: T
  readonly options: readonly ChoiceOption<T>[]
  readonly disabled?: boolean
  /** Shown while there is nothing to choose. */
  readonly empty?: string
  readonly onChange: (value: T) => void
}): ReactElement {
  const [filter, setFilter] = useState('')
  const needle = filter.trim().toLowerCase()
  const shown = needle === '' ? options : options.filter(option => `${option.label} ${option.hint ?? ''}`.toLowerCase().includes(needle))
  return (
    <div className="xlist">
      {options.length > LIST_FILTER_AT ? (
        <input type="search" className="xlist__filter" aria-label={`Filter ${label}`} placeholder="Filter…" value={filter} onChange={event => setFilter(event.target.value)} />
      ) : null}
      <div className="xlist__rows" role="radiogroup" aria-label={label}>
        {shown.length === 0 ? <p className="xlist__empty">{options.length === 0 ? empty ?? 'Nothing to choose yet.' : 'No match.'}</p> : shown.map(option => {
          const on = option.value === value
          return (
            <button
              key={option.value}
              type="button"
              role="radio"
              aria-checked={on}
              className={`xlist__row${on ? ' is-on' : ''}`}
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
