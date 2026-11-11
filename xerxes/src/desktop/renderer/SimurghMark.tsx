// Licensed under the Apache License, Version 2.0.

import type { ReactElement } from 'react'

/** The Xerxes Simurgh: a small geometric bird mark, drawn entirely locally. */
export function SimurghMark({ size = 24, className = '' }: { readonly size?: number; readonly className?: string }): ReactElement {
  return (
    <svg className={`simurgh-mark${className ? ` ${className}` : ''}`} width={size} height={size} viewBox="0 0 64 64" fill="none" aria-hidden="true">
      <path d="M10 39.5c10.6-1.8 16.2-8.2 21.2-15.8-1.6-6.5 2.5-12.1 10.3-14.7-2.1 3.3-1.8 6.5.2 9.2 6.4-.7 11.1 1.7 14.1 6.8-8.2-1.9-13.3.8-15.5 6.4-3.1 7.7-9.3 12.4-18.7 13.8l-6.3 8.3 1.8-9.1-8.5 3.8 4.6-8.1-3.1-1.6Z" fill="currentColor" />
      <path d="M20.2 37.8c8.9-1.1 15.2-5.4 19-12.8-7.6 2.1-13.5 1-17.7-3.4 7.8-.1 13.2-2.4 16.2-6.8-10.9 2.2-18.1 9.9-17.5 23Z" fill="currentColor" opacity=".68" />
      <path d="M29.2 45.2c-2.2 5.1-6.1 8.8-11.8 11.1 2.1-3.5 2.8-7 2.1-10.5l9.7-.6Z" fill="currentColor" opacity=".82" />
      <circle cx="43.3" cy="17.2" r="1.35" fill="var(--x-screen, #111)" />
    </svg>
  )
}
