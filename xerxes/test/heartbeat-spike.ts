// temporary spike to observe timer behavior during sync loops
let count = 0
const interval = setInterval(() => { count += 1 }, 10)
const start = Date.now()
while (Date.now() - start < 100) {
  // busy wait
}
clearInterval(interval)
console.log(count)
