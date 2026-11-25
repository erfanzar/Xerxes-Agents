// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Sleek, theme-aware custom select control matching Xerxes's Vibes / Immortals
 * popover grammar (`ScrollableCombobox` / `SideQuestionModelPicker`) — replacing
 * jarring native OS `<select>` menus across Settings.
 */

import {
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
  type ReactElement,
} from 'react'
import { createPortal } from 'react-dom'

import { Icon } from './Icon.js'
import { computeComboboxPlacement, type ComboboxPlacement } from './ScrollableCombobox.js'

export interface SelectFieldOption {
  readonly value: string
  readonly label: string
  readonly detail?: string
  readonly disabled?: boolean
}

export interface SelectFieldProps {
  readonly id?: string
  readonly value: string
  readonly options: readonly SelectFieldOption[]
  readonly onChange: (value: string) => void
  readonly placeholder?: string
  readonly ariaLabel?: string
  readonly ariaLabelledBy?: string
  readonly ariaInvalid?: boolean
  readonly ariaDescribedBy?: string
  readonly listLabel: string
  readonly disabled?: boolean
  readonly className?: string
  readonly maxHeight?: number
}

export function SelectField({
  id,
  value,
  options,
  onChange,
  placeholder = 'Select…',
  ariaLabel,
  ariaLabelledBy,
  ariaInvalid,
  ariaDescribedBy,
  listLabel,
  disabled = false,
  className,
  maxHeight = 280,
}: SelectFieldProps): ReactElement {
  const generatedId = useId()
  const listId = `${id ?? generatedId}-listbox`
  const triggerRef = useRef<HTMLButtonElement>(null)
  const popupRef = useRef<HTMLDivElement>(null)
  const listRef = useRef<HTMLDivElement>(null)
  const scrollActiveIntoViewRef = useRef(false)

  const [open, setOpen] = useState(false)
  const [activeIndex, setActiveIndex] = useState(0)
  const [placement, setPlacement] = useState<ComboboxPlacement | null>(null)

  const selectedOption = options.find(opt => opt.value === value)
  const showPopup = !disabled && open && options.length > 0

  const firstEnabledIndex = (from = 0, step = 1): number => {
    const len = options.length
    if (len === 0) return -1
    for (let i = 0; i < len; i += 1) {
      const idx = ((from + i * step) % len + len) % len
      if (!options[idx]?.disabled) return idx
    }
    return -1
  }

  useLayoutEffect(() => {
    if (!showPopup || typeof window === 'undefined') return
    const place = (): void => {
      const anchorEl = triggerRef.current
      if (!anchorEl) return
      const rect = anchorEl.getBoundingClientRect()
      setPlacement(
        computeComboboxPlacement(
          { left: rect.left, top: rect.top, bottom: rect.bottom, width: rect.width },
          { width: window.innerWidth, height: window.innerHeight },
          maxHeight,
        ),
      )
    }
    place()
    const observer = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(place) : null
    if (triggerRef.current) observer?.observe(triggerRef.current)
    const onScroll = (event: Event): void => {
      const target = event.target
      if (target instanceof Node && popupRef.current?.contains(target)) return
      place()
    }
    window.addEventListener('resize', place)
    window.addEventListener('scroll', onScroll, true)
    return () => {
      observer?.disconnect()
      window.removeEventListener('resize', place)
      window.removeEventListener('scroll', onScroll, true)
    }
  }, [showPopup, maxHeight, options.length])

  useEffect(() => {
    if (!showPopup || typeof document === 'undefined') return
    const onPointerDown = (event: MouseEvent): void => {
      const target = event.target
      if (!(target instanceof Node)) return
      if (triggerRef.current?.contains(target) || popupRef.current?.contains(target)) return
      setOpen(false)
    }
    document.addEventListener('mousedown', onPointerDown, true)
    return () => document.removeEventListener('mousedown', onPointerDown, true)
  }, [showPopup])

  useEffect(() => {
    if (!showPopup || typeof window === 'undefined') return
    const onEscape = (event: globalThis.KeyboardEvent): void => {
      if (event.key !== 'Escape') return
      event.preventDefault()
      event.stopPropagation()
      setOpen(false)
      triggerRef.current?.focus()
    }
    window.addEventListener('keydown', onEscape, true)
    return () => window.removeEventListener('keydown', onEscape, true)
  }, [showPopup])

  useEffect(() => {
    if (!showPopup) return
    setActiveIndex(prev => (prev >= 0 && prev < options.length ? prev : 0))
  }, [showPopup, options.length])

  useEffect(() => {
    if (!showPopup || !scrollActiveIntoViewRef.current) return
    scrollActiveIntoViewRef.current = false
    const listEl = listRef.current
    if (!listEl) return
    const activeEl = listEl.querySelector<HTMLElement>(`[data-option-index="${activeIndex}"]`)
    activeEl?.scrollIntoView?.({ block: 'nearest' })
  }, [showPopup, activeIndex, options.length])

  const openMenu = (): void => {
    if (disabled || options.length === 0) return
    const selectedIdx = options.findIndex(opt => opt.value === value && !opt.disabled)
    const initial = selectedIdx >= 0 ? selectedIdx : firstEnabledIndex(0, 1)
    scrollActiveIntoViewRef.current = true
    setActiveIndex(initial >= 0 ? initial : 0)
    setOpen(true)
  }

  const pickOption = (opt: SelectFieldOption): void => {
    if (opt.disabled) return
    onChange(opt.value)
    setOpen(false)
    triggerRef.current?.focus()
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLButtonElement>): void => {
    if (disabled || options.length === 0) return
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      if (!showPopup) {
        openMenu()
        return
      }
      const next = firstEnabledIndex(activeIndex + 1, 1)
      if (next >= 0) {
        scrollActiveIntoViewRef.current = true
        setActiveIndex(next)
      }
      return
    }
    if (event.key === 'ArrowUp') {
      event.preventDefault()
      if (!showPopup) {
        openMenu()
        return
      }
      const next = firstEnabledIndex(activeIndex - 1, -1)
      if (next >= 0) {
        scrollActiveIntoViewRef.current = true
        setActiveIndex(next)
      }
      return
    }
    if (event.key === 'Home' && showPopup) {
      event.preventDefault()
      const next = firstEnabledIndex(0, 1)
      if (next >= 0) {
        scrollActiveIntoViewRef.current = true
        setActiveIndex(next)
      }
      return
    }
    if (event.key === 'End' && showPopup) {
      event.preventDefault()
      const next = firstEnabledIndex(options.length - 1, -1)
      if (next >= 0) {
        scrollActiveIntoViewRef.current = true
        setActiveIndex(next)
      }
      return
    }
    if ((event.key === 'Enter' || event.key === ' ') && showPopup) {
      event.preventDefault()
      const chosen = options[activeIndex]
      if (chosen && !chosen.disabled) pickOption(chosen)
      return
    }
    if (event.key === 'Tab' && showPopup) {
      setOpen(false)
      return
    }
    if (event.key === 'Escape' && showPopup) {
      event.preventDefault()
      event.stopPropagation()
      setOpen(false)
      return
    }
    if (showPopup && event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey) {
      const ch = event.key.toLowerCase()
      const len = options.length
      for (let i = 1; i <= len; i += 1) {
        const idx = (activeIndex + i) % len
        const opt = options[idx]
        if (opt && !opt.disabled && opt.label.toLowerCase().startsWith(ch)) {
          scrollActiveIntoViewRef.current = true
          setActiveIndex(idx)
          break
        }
      }
    }
  }

  const activeOptionId =
    showPopup && activeIndex >= 0 && activeIndex < options.length
      ? `${listId}-opt-${activeIndex}`
      : undefined

  const popupStyle: CSSProperties = placement
    ? placement.placement === 'below'
      ? {
          left: `${placement.left}px`,
          width: `${placement.width}px`,
          top: `${placement.top}px`,
          bottom: 'auto',
          maxHeight: `${placement.maxHeight}px`,
        }
      : {
          left: `${placement.left}px`,
          width: `${placement.width}px`,
          top: 'auto',
          bottom: `${placement.bottom}px`,
          maxHeight: `${placement.maxHeight}px`,
        }
    : { maxHeight: `${maxHeight}px` }

  const popupNode = showPopup ? (
    <div
      ref={popupRef}
      className="combo-pop select-pop"
      data-placement={placement?.placement ?? 'below'}
      style={popupStyle}
    >
      <div
        ref={listRef}
        id={listId}
        className="combo-pop__list"
        role="listbox"
        aria-label={listLabel}
      >
        {options.map((opt, index) => {
          const isSelected = opt.value === value
          const isActive = index === activeIndex
          return (
            <button
              key={`${opt.value}:${index}`}
              id={`${listId}-opt-${index}`}
              type="button"
              role="option"
              data-option-index={index}
              aria-selected={isSelected}
              aria-disabled={opt.disabled || undefined}
              disabled={opt.disabled}
              className={`combo-pop__option select-pop__option${isActive ? ' is-active' : ''}${isSelected ? ' is-selected' : ''}`}
              onMouseDown={event => event.preventDefault()}
              onMouseEnter={() => {
                if (!opt.disabled) setActiveIndex(index)
              }}
              onClick={() => pickOption(opt)}
            >
              <span className="select-pop__text">
                <span className="combo-pop__label">{opt.label}</span>
                {opt.detail && <span className="select-pop__detail">{opt.detail}</span>}
              </span>
              {isSelected && (
                <span className="combo-pop__check" aria-hidden="true">
                  <Icon name="check" size={12} />
                </span>
              )}
            </button>
          )
        })}
      </div>
    </div>
  ) : null

  return (
    <div className={`select-field${showPopup ? ' is-open' : ''}${disabled ? ' is-disabled' : ''}${className ? ` ${className}` : ''}`}>
      <button
        ref={triggerRef}
        {...(id !== undefined ? { id } : {})}
        type="button"
        role="combobox"
        aria-haspopup="listbox"
        aria-expanded={showPopup}
        {...(showPopup ? { 'aria-controls': listId } : {})}
        {...(activeOptionId !== undefined ? { 'aria-activedescendant': activeOptionId } : {})}
        {...(ariaLabel !== undefined ? { 'aria-label': ariaLabel } : {})}
        {...(ariaLabelledBy !== undefined ? { 'aria-labelledby': ariaLabelledBy } : {})}
        {...(ariaInvalid ? { 'aria-invalid': true } : {})}
        {...(ariaDescribedBy !== undefined ? { 'aria-describedby': ariaDescribedBy } : {})}
        disabled={disabled}
        className="select-field__trigger"
        onClick={() => {
          if (showPopup) setOpen(false)
          else openMenu()
        }}
        onKeyDown={handleKeyDown}
      >
        <span className={`select-field__value${!selectedOption ? ' is-placeholder' : ''}`}>
          {selectedOption ? selectedOption.label : placeholder}
        </span>
        <span className="select-field__caret" aria-hidden="true">
          <Icon name="caretDown" size={11} />
        </span>
      </button>
      {popupNode && (
        typeof document === 'undefined' || !document.body
          ? popupNode
          : createPortal(<div className="atelier picker-layer">{popupNode}</div>, document.body)
      )}
    </div>
  )
}
