import { writeFileSync, rmSync, mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
const dir = mkdtempSync(join(tmpdir(), 'hb-'))
let count = 0
const interval = setInterval(() => { count += 1 }, 10)
for (let i = 0; i < 100; i += 1) {
  const path = join(dir, String(i))
  writeFileSync(path, 'x')
  rmSync(path)
}
clearInterval(interval)
rmSync(dir, { recursive: true, force: true })
console.log(count)
