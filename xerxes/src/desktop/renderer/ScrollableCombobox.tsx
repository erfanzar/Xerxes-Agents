// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Scrollable, theme-aware combobox replacing native HTML `<datalist>` controls.
 *
 * Chromium/Electron's native `<datalist>` popup overflows viewport bounds and
 * refuses to scroll when lists grow past ~15–20 entries (such as `~/.ssh/config`
 * hosts or full provider model catalogs) and ignores dark/custom themes.
 * This component renders a portal-mounted, viewport-clamped `role="listbox"`
 * with smooth wheel/trackpad/scrollbar scrolling, live filtering, and full
 * keyboard navigation while keeping the input freely editable.
 */

import {
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
  type ReactElement,
} from 'react'
import { createPortal } from 'react-dom'

import { Icon } from './Icon.js'

export interface ComboboxAnchorRect {
  readonly left: number
  readonly top: number
  readonly bottom: number
  readonly width: number
}

export interface ComboboxViewport {
  readonly width: number
  readonly height: number
}

export type ComboboxPlacement =
  | {
      readonly placement: 'below'
      readonly left: number
      readonly width: number
      readonly top: number
      readonly maxHeight: number
    }
  | {
      readonly placement: 'above'
      readonly left: number
      readonly width: number
      readonly bottom: number
      readonly maxHeight: number
    }

/**
 * Deduplicate and optionally filter combobox options.
 * When `filtering` is false (e.g. clicking the chevron or focusing an already
 * selected value), all options stay visible so the user can browse the entire
 * list. When `filtering` is true, prefix matches rank ahead of substring matches.
 */
export function filterComboboxOptions(
  options: readonly string[],
  query: string,
  filtering: boolean,
): string[] {
  const unique: string[] = []
  const seen = new Set<string>()
  for (const option of options) {
    if (!option || seen.has(option)) continue
    seen.add(option)
    unique.push(option)
  }
  const needle = query.trim().toLowerCase()
  if (!filtering || !needle) return unique
  const prefix: string[] = []
  const contains: string[] = []
  for (const option of unique) {
    const lower = option.toLowerCase()
    if (lower.startsWith(needle)) prefix.push(option)
    else if (lower.includes(needle)) contains.push(option)
  }
  return [...prefix, ...contains]
}

/**
 * Compute viewport-clamped fixed coordinates and bounded height for a combobox
 * popover so long lists never clip off the top or bottom edge of the window.
 */
export function computeComboboxPlacement(
  anchor: ComboboxAnchorRect,
  viewport: ComboboxViewport,
  preferredMaxHeight = 264,
): ComboboxPlacement {
  const margin = 12
  const gap = 6
  const minUsableHeight = 96
  const spaceBelow = Math.max(0, viewport.height - anchor.bottom - margin - gap)
  const spaceAbove = Math.max(0, anchor.top - margin - gap)
  const preferBelow = spaceBelow >= Math.min(180, preferredMaxHeight) || spaceBelow >= spaceAbove
  const available = preferBelow ? spaceBelow : spaceAbove
  const maxHeight = Math.max(minUsableHeight, Math.min(preferredMaxHeight, Math.floor(available)))
  const width = Math.max(160, Math.min(Math.round(anchor.width), Math.max(160, viewport.width - margin * 2)))
  const left = Math.max(margin, Math.min(Math.round(anchor.left), Math.max(margin, viewport.width - width - margin)))

  if (preferBelow) {
    return {
      placement: 'below',
      left,
      width,
      top: Math.round(anchor.bottom + gap),
      maxHeight,
    }
  }
  return {
    placement: 'above',
    left,
    width,
    bottom: Math.round(viewport.height - anchor.top + gap),
    maxHeight,
  }
}

export interface ScrollableComboboxProps {
  readonly id?: string
  readonly value: string
  readonly options: readonly string[]
  readonly onChange: (value: string) => void
  readonly onSelect?: (value: string) => void
  readonly placeholder?: string
  readonly ariaLabel?: string
  readonly ariaLabelledBy?: string
  readonly ariaInvalid?: boolean
  readonly ariaDescribedBy?: string
  readonly listLabel: string
  readonly toggleLabel?: string
  readonly spellCheck?: boolean
  readonly disabled?: boolean
  readonly variant?: 'workspace' | 'field'
  readonly className?: string
  readonly maxHeight?: number
}

export function ScrollableCombobox({
  id,
  value,
  options,
  onChange,
  onSelect,
  placeholder,
  ariaLabel,
  ariaLabelledBy,
  ariaInvalid,
  ariaDescribedBy,
  listLabel,
  toggleLabel,
  spellCheck = false,
  disabled = false,
  variant = 'field',
  className,
  maxHeight = 264,
}: ScrollableComboboxProps): ReactElement {
  const generatedId = useId()
  const listId = `${id ?? generatedId}-listbox`
  const wrapperRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const popupRef = useRef<HTMLDivElement>(null)
  const listRef = useRef<HTMLDivElement>(null)
  const scrollActiveIntoViewRef = useRef(false)
  const suppressNextFocusOpenRef = useRef(false)
  const keyboardNavigatedRef = useRef(false)

  const [open, setOpen] = useState(false)
  const [filtering, setFiltering] = useState(false)
  const [activeIndex, setActiveIndex] = useState(0)
  const [placement, setPlacement] = useState<ComboboxPlacement | null>(null)

  const allOptions = useMemo(() => filterComboboxOptions(options, '', false), [options])
  const visibleOptions = useMemo(
    () => filterComboboxOptions(options, value, filtering),
    [options, value, filtering],
  )
  const hasOptions = allOptions.length > 0
  const showPopup = !disabled && open && visibleOptions.length > 0

  useEffect(() => {
    if (!showPopup) return
    setActiveIndex(prev => {
      if (prev >= 0 && prev < visibleOptions.length) return prev
      return 0
    })
  }, [showPopup, visibleOptions.length])

  // Position popup against the input wrapper and track viewport/ancestor scrolls
  // while ignoring scrolls inside the dropdown's own listbox.
  useLayoutEffect(() => {
    if (!showPopup || typeof window === 'undefined') return
    const place = (): void => {
      const anchorEl = wrapperRef.current
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
    if (wrapperRef.current) observer?.observe(wrapperRef.current)
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
  }, [showPopup, maxHeight, visibleOptions.length])

  // Close on outside click without blocking scrollbar interaction inside popupRef.
  useEffect(() => {
    if (!showPopup || typeof document === 'undefined') return
    const onPointerDown = (event: MouseEvent): void => {
      const target = event.target
      if (!(target instanceof Node)) return
      if (wrapperRef.current?.contains(target) || popupRef.current?.contains(target)) return
      setOpen(false)
      setFiltering(false)
    }
    document.addEventListener('mousedown', onPointerDown, true)
    return () => document.removeEventListener('mousedown', onPointerDown, true)
  }, [showPopup])

  // Capture Escape so closing the combobox never dismisses an enclosing dialog sheet,
  // and close the popup when the window loses focus.
  useEffect(() => {
    if (!showPopup || typeof window === 'undefined') return
    const onEscape = (event: globalThis.KeyboardEvent): void => {
      if (event.key !== 'Escape') return
      event.preventDefault()
      event.stopPropagation()
      setOpen(false)
      setFiltering(false)
      keyboardNavigatedRef.current = false
      suppressNextFocusOpenRef.current = true
      inputRef.current?.focus()
    }
    const onWindowBlur = (): void => {
      setOpen(false)
      setFiltering(false)
      keyboardNavigatedRef.current = false
    }
    window.addEventListener('keydown', onEscape, true)
    window.addEventListener('blur', onWindowBlur)
    return () => {
      window.removeEventListener('keydown', onEscape, true)
      window.removeEventListener('blur', onWindowBlur)
    }
  }, [showPopup])

  // Scroll highlighted option into view on open and keyboard navigation.
  useEffect(() => {
    if (!showPopup || !scrollActiveIntoViewRef.current) return
    scrollActiveIntoViewRef.current = false
    const listEl = listRef.current
    if (!listEl) return
    const activeEl = listEl.querySelector<HTMLElement>(`[data-option-index="${activeIndex}"]`)
    activeEl?.scrollIntoView?.({ block: 'nearest' })
  }, [showPopup, activeIndex, visibleOptions.length])

  const openFullList = (): void => {
    if (disabled || !hasOptions) return
    const currentIndex = allOptions.findIndex(opt => opt.toLowerCase() === value.trim().toLowerCase())
    scrollActiveIntoViewRef.current = true
    keyboardNavigatedRef.current = false
    setFiltering(false)
    setActiveIndex(currentIndex >= 0 ? currentIndex : 0)
    setOpen(true)
  }

  const selectOption = (option: string): void => {
    if (onSelect) onSelect(option)
    else onChange(option)
    setOpen(false)
    setFiltering(false)
    keyboardNavigatedRef.current = false
    suppressNextFocusOpenRef.current = true
    inputRef.current?.focus()
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>): void => {
    if (disabled || !hasOptions) return
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      if (!showPopup) {
        openFullList()
        keyboardNavigatedRef.current = true
        return
      }
      scrollActiveIntoViewRef.current = true
      keyboardNavigatedRef.current = true
      setActiveIndex(prev => (prev + 1) % visibleOptions.length)
      return
    }
    if (event.key === 'ArrowUp') {
      event.preventDefault()
      if (!showPopup) {
        scrollActiveIntoViewRef.current = true
        keyboardNavigatedRef.current = true
        setFiltering(false)
        setActiveIndex(allOptions.length - 1)
        setOpen(true)
        return
      }
      scrollActiveIntoViewRef.current = true
      keyboardNavigatedRef.current = true
      setActiveIndex(prev => (prev - 1 + visibleOptions.length) % visibleOptions.length)
      return
    }
    if (event.key === 'PageDown' && showPopup) {
      event.preventDefault()
      scrollActiveIntoViewRef.current = true
      keyboardNavigatedRef.current = true
      setActiveIndex(prev => Math.min(visibleOptions.length - 1, prev + 6))
      return
    }
    if (event.key === 'PageUp' && showPopup) {
      event.preventDefault()
      scrollActiveIntoViewRef.current = true
      keyboardNavigatedRef.current = true
      setActiveIndex(prev => Math.max(0, prev - 6))
      return
    }
    if (event.key === 'Home' && showPopup) {
      event.preventDefault()
      scrollActiveIntoViewRef.current = true
      keyboardNavigatedRef.current = true
      setActiveIndex(0)
      return
    }
    if (event.key === 'End' && showPopup) {
      event.preventDefault()
      scrollActiveIntoViewRef.current = true
      keyboardNavigatedRef.current = true
      setActiveIndex(visibleOptions.length - 1)
      return
    }
    if (event.key === 'Enter' && showPopup) {
      const chosen = visibleOptions[activeIndex]
      const exactMatch = visibleOptions.find(opt => opt.toLowerCase() === value.trim().toLowerCase())
      if (keyboardNavigatedRef.current && chosen !== undefined) {
        event.preventDefault()
        selectOption(chosen)
      } else if (exactMatch !== undefined) {
        event.preventDefault()
        selectOption(exactMatch)
      } else {
        setOpen(false)
        setFiltering(false)
      }
      return
    }
    if (event.key === 'Escape' && showPopup) {
      event.preventDefault()
      event.stopPropagation()
      setOpen(false)
      setFiltering(false)
      keyboardNavigatedRef.current = false
    }
  }

  const normalizedValue = value.trim().toLowerCase()
  const activeOptionId = showPopup && activeIndex >= 0 && activeIndex < visibleOptions.length
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
      className={`combo-pop${variant === 'workspace' ? ' combo-pop--workspace' : ''}`}
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
        {visibleOptions.map((option, index) => {
          const isSelected = option.toLowerCase() === normalizedValue
          const isActive = index === activeIndex
          return (
            <button
              key={option}
              id={`${listId}-opt-${index}`}
              type="button"
              role="option"
              data-option-index={index}
              aria-selected={isSelected}
              className={`combo-pop__option${isActive ? ' is-active' : ''}${isSelected ? ' is-selected' : ''}`}
              onMouseDown={event => event.preventDefault()}
              onMouseEnter={() => setActiveIndex(index)}
              onClick={() => selectOption(option)}
            >
              <span className="combo-pop__label">{option}</span>
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
    <div
      ref={wrapperRef}
      className={`combo-field combo-field--${variant}${hasOptions ? ' has-options' : ''}${showPopup ? ' is-open' : ''}${className ? ` ${className}` : ''}`}
    >
      <input
        ref={inputRef}
        {...(id !== undefined ? { id } : {})}
        type="text"
        role="combobox"
        aria-autocomplete="list"
        aria-expanded={showPopup}
        {...(showPopup ? { 'aria-controls': listId } : {})}
        {...(activeOptionId !== undefined ? { 'aria-activedescendant': activeOptionId } : {})}
        {...(ariaLabel !== undefined ? { 'aria-label': ariaLabel } : {})}
        {...(ariaLabelledBy !== undefined ? { 'aria-labelledby': ariaLabelledBy } : {})}
        {...(ariaInvalid ? { 'aria-invalid': true } : {})}
        {...(ariaDescribedBy !== undefined ? { 'aria-describedby': ariaDescribedBy } : {})}
        value={value}
        {...(placeholder !== undefined ? { placeholder } : {})}
        spellCheck={spellCheck}
        disabled={disabled}
        autoComplete="off"
        className="combo-field__input"
        onFocus={() => {
          if (suppressNextFocusOpenRef.current) {
            suppressNextFocusOpenRef.current = false
            return
          }
          if (hasOptions) openFullList()
        }}
        onClick={() => {
          if (hasOptions && !open) openFullList()
        }}
        onChange={event => {
          onChange(event.target.value)
          if (hasOptions) {
            keyboardNavigatedRef.current = false
            setFiltering(true)
            setActiveIndex(0)
            setOpen(true)
          }
        }}
        onBlur={event => {
          const next = event.relatedTarget
          if (next instanceof Node && !wrapperRef.current?.contains(next) && !popupRef.current?.contains(next)) {
            setOpen(false)
            setFiltering(false)
          }
        }}
        onKeyDown={handleKeyDown}
      />
      {hasOptions && (
        <button
          type="button"
          className="combo-field__toggle"
          tabIndex={-1}
          disabled={disabled}
          aria-label={toggleLabel ?? `Toggle ${listLabel}`}
          aria-expanded={showPopup}
          onMouseDown={event => event.preventDefault()}
          onClick={() => {
            if (showPopup) {
              setOpen(false)
              setFiltering(false)
            } else {
              openFullList()
              suppressNextFocusOpenRef.current = true
              inputRef.current?.focus()
            }
          }}
        >
          <Icon name="caretDown" size={11} />
        </button>
      )}
      {popupNode && (
        typeof document === 'undefined' || !document.body
          ? popupNode
          : createPortal(<div className="atelier picker-layer">{popupNode}</div>, document.body)
      )}
    </div>
  )
}
