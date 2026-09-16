// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useState, type ReactElement } from 'react'
import { Icon } from './Icon.js'

import type { TodoItem } from "./todoState.js"
export { todoItemsOf, todosFromResult, type TodoItem } from "./todoState.js"

export function TodoList({ items }: { items: readonly TodoItem[] }): ReactElement | null {
  const [open, setOpen] = useState(true)
  if (items.length === 0) return null
  const done = items.filter(item=>item.status==='completed').length
  return <section className="todos" aria-label="Session todos">
    <button className="todos__head" aria-expanded={open} onClick={()=>setOpen(value=>!value)}><Icon name="plan" size={14}/><span className="todos__title">To-dos</span><span className="todos__counts">{done}/{items.length} completed</span><Icon name="chevron" size={12}/></button>
    {open && <div className="todos__list">{items.map((item,index)=><div key={`${item.id}:${index}`} className={`todo todo--${item.status==='completed'?'done':item.status==='in_progress'?'cur':'todo'}`}><span className="todo__t">{item.content}</span><span className="todo__state">{item.status==='in_progress'?'In progress':item.status==='completed'?'Completed':'Pending'}</span></div>)}</div>}
  </section>
}
