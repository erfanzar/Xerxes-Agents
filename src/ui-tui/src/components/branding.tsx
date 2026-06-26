import { Box, Text, useStdout } from '@xerxes/ink'

import { artWidth, caduceus, CADUCEUS_WIDTH, logo, LOGO_WIDTH } from '../banner.js'
import type { Theme } from '../theme.js'
import type { PanelSection, SessionInfo } from '../types.js'

const HIDE_BELOW = 34
const COMPACT_FROM = 58
const BANNER_COMMANDS: [string, string][] = [
  ['/init', 'set up a project'],
  ['/provider', 'pick a model'],
  ['/resume', 'reopen a session'],
  ['/skin', 'theme'],
  ['/help', '']
]
const FEATURED_SKILLS = ['eternal-army', 'xerxes-agent', 'plan', 'deepscan', 'autoresearch', 'systematic-debugging']

export function ArtLines({ lines }: { lines: [string, string][] }) {
  return (
    <Box flexDirection="column" height={lines.length} opaque width={artWidth(lines)}>
      {lines.map(([c, text], i) => (
        <Text color={c} key={i} wrap="truncate-end">
          {text}
        </Text>
      ))}
    </Box>
  )
}

const clip = (s: string, w: number) => (w <= 0 ? '' : s.length > w ? `${s.slice(0, Math.max(0, w - 1))}…` : s)

const centerIn = (s: string, w: number) => {
  const f = clip(s, w)
  const slack = Math.max(0, w - f.length)
  const left = slack >> 1

  return `${' '.repeat(left)}${f}${' '.repeat(slack - left)}`
}

const ruleIn = (label: string, w: number) => {
  const f = clip(label, Math.max(1, w - 4))
  const slack = Math.max(0, w - f.length - 2)
  const left = slack >> 1

  return `${'─'.repeat(left)} ${f} ${'─'.repeat(slack - left)}`
}

function CompactBanner({ cols, t }: { cols: number; t: Theme }) {
  const w = Math.max(28, cols - 4)

  return (
    <Box flexDirection="column" height={3} marginBottom={1} opaque width={w}>
      <Text bold color={t.color.primary}>
        {ruleIn(t.brand.name, w)}
      </Text>
      <Text color={t.color.muted}>{centerIn(t.brand.welcome, w)}</Text>
      <Text color={t.color.primary}>{'─'.repeat(w)}</Text>
    </Box>
  )
}

export function Banner({ maxWidth, t }: { maxWidth?: number; t: Theme }) {
  const term = useStdout().stdout?.columns ?? 80
  const cols = Math.max(1, Math.min(term, maxWidth ?? term))

  if (cols < HIDE_BELOW) {
    return null
  }

  const logoLines = logo(t.color, t.bannerLogo || undefined)
  const logoW = t.bannerLogo ? artWidth(logoLines) : LOGO_WIDTH

  if (cols >= logoW + 2) {
    return (
      <Box flexDirection="column" marginBottom={1}>
        <ArtLines lines={logoLines} />
      </Box>
    )
  }

  if (cols >= COMPACT_FROM) {
    return <CompactBanner cols={cols} t={t} />
  }

  const name = cols >= 52 ? t.brand.name : (t.brand.name.split(' ')[0] ?? t.brand.name)

  return (
    <Box flexDirection="column" marginBottom={1}>
      <Text bold color={t.color.primary} wrap="truncate-end">
        {t.brand.icon} {name}
      </Text>
      <Text color={t.color.muted} wrap="truncate-end">
        {t.brand.welcome}
      </Text>
    </Box>
  )
}

function flattenRecord(record: Record<string, string[]>): string[] {
  return Object.values(record).flat()
}

function orderedSkills(record: Record<string, string[]>): string[] {
  const seen = new Set<string>()
  const all = flattenRecord(record)
    .filter(skill => {
      if (!skill || seen.has(skill)) {
        return false
      }
      seen.add(skill)

      return true
    })
    .sort()

  return [...FEATURED_SKILLS.filter(skill => seen.has(skill)), ...all.filter(skill => !FEATURED_SKILLS.includes(skill))]
}

function shortHome(path: string): string {
  const home = process.env.HOME

  return home && path.startsWith(home) ? `~${path.slice(home.length)}` : path
}

function initialProjectDir(): string {
  return process.env.XERXES_PROJECT_DIR || process.env.XERXES_CWD || process.cwd()
}

export function SessionPanel({ info, maxWidth, sid, t }: SessionPanelProps) {
  const term = useStdout().stdout?.columns ?? 100
  const cols = Math.max(20, Math.min(term, maxWidth ?? term))
  const heroLines = caduceus(t.color, t.bannerHero || undefined)
  const leftW = artWidth(heroLines) || CADUCEUS_WIDTH
  const wide = cols >= 86 && leftW + 36 < cols
  const railTargetW = cols >= 150 ? Math.floor((cols - leftW - 10) * 0.56) : 0
  const infoW = Math.max(24, wide ? Math.min(46, Math.max(34, cols - leftW - railTargetW - 10)) : cols - 8)
  const skills = orderedSkills(info.skills)
  const skillsTotal = flattenRecord(info.skills).length
  const toolsTotal = flattenRecord(info.tools).length
  const mcpConnected = (info.mcp_servers ?? []).filter(s => s.connected).length
  const updateText =
    typeof info.update_behind === 'number' && info.update_behind > 0
      ? `${info.update_behind} ahead available`
      : 'current'
  const rows: [string, string][] = [
    ['Directory:', shortHome(info.cwd || initialProjectDir())],
    ['Version:', info.version ? `v${info.version}` : '(unknown)'],
    ['Session:', sid || '(new)'],
    ['Model:', info.model || '(not set - pick one with /provider)'],
    ['HEAD:', info.head_hash || '(unknown)'],
    ['Updates:', updateText]
  ]
  const labelW = Math.max(...rows.map(([label]) => label.length))
  const maxRailRows = Math.max(0, heroLines.length - 4)
  const railW = Math.max(0, cols - leftW - infoW - 12)
  const showRail = wide && railW >= 34
  const railCols = railW >= 82 ? 2 : 1
  const skillCellW = railCols > 1 ? Math.floor((railW - 2) / 2) : railW
  const skillCapacity = Math.max(0, maxRailRows * railCols)
  const shownSkills = skills.slice(0, skillCapacity)
  const skillRows = Math.ceil(shownSkills.length / railCols)
  const skillsLabel = skillsTotal > 0 ? `${skillsTotal} skills` : ''

  return (
    <Box
      borderColor={t.color.border}
      borderStyle="round"
      marginBottom={1}
      paddingX={2}
      paddingY={1}
      width={Math.max(28, cols - 2)}
    >
      {wide && (
        <Box flexDirection="column" marginRight={3} width={leftW}>
          <ArtLines lines={heroLines} />
        </Box>
      )}

      <Box flexDirection="column" width={infoW}>
        {!wide && (
          <Box flexDirection="column" marginBottom={1}>
            <ArtLines lines={heroLines} />
          </Box>
        )}

        {rows.map(([label, value]) => (
          <Text key={label} wrap="truncate-end">
            <Text color={t.color.sessionLabel}>{label.padEnd(labelW)}</Text>
            {'  '}
            <Text color={t.color.text}>{value}</Text>
          </Text>
        ))}

        <Text />
        <Text bold color={t.color.system}>
          {t.brand.welcome}
        </Text>
        <Text color={t.color.muted}>Send /help for help information.</Text>

        {(toolsTotal > 0 || skillsTotal > 0 || mcpConnected > 0) && (
          <Text color={t.color.muted} wrap="truncate-end">
            {`${toolsTotal} tools · ${skillsTotal} skills${mcpConnected ? ` · ${mcpConnected} MCP` : ''}`}
          </Text>
        )}
      </Box>

      {showRail && (
        <Box flexDirection="column" marginLeft={2} width={railW}>
          <Text bold color={t.color.system} wrap="truncate-end">
            {t.brand.helpHeader}
            {skillsLabel ? <Text color={t.color.muted}> {skillsLabel}</Text> : null}
          </Text>
          <Text color={t.color.muted} wrap="truncate-end">
            commands{'  '}
            {BANNER_COMMANDS.map(([name, desc], i) => (
              <Text key={name}>
                {i > 0 ? '  ' : ''}
                <Text color={t.color.warn}>{name}</Text>
                {desc ? <Text color={t.color.muted}> {desc}</Text> : null}
              </Text>
            ))}
          </Text>
          <Text />
          {!info.lazy && shownSkills.length === 0 && (
            <Text color={t.color.muted} wrap="truncate-end">
              skills not loaded yet — /skills
            </Text>
          )}
          {Array.from({ length: skillRows }).map((_, row) => (
            <Box key={row}>
              {Array.from({ length: railCols }).map((__, col) => {
                const skill = shownSkills[row * railCols + col]

                if (!skill) {
                  return <Box key={col} width={skillCellW} />
                }

                const desc = info.skillDescriptions?.[skill] ?? ''
                const featured = FEATURED_SKILLS.includes(skill)

                return (
                  <Box key={skill} marginRight={col < railCols - 1 ? 2 : 0} width={skillCellW}>
                    <Text wrap="truncate-end">
                      <Text color={featured ? t.color.warn : t.color.muted}>{featured ? '★ ' : '  '}</Text>
                      <Text color={t.color.accent}>/{skill}</Text>
                      {desc ? (
                        <Text color={t.color.muted}> {clip(desc, Math.max(0, skillCellW - skill.length - 5))}</Text>
                      ) : null}
                    </Text>
                  </Box>
                )
              })}
            </Box>
          ))}
          {skills.length > shownSkills.length && (
            <Text color={t.color.muted}>+ {skills.length - shownSkills.length} more — /help</Text>
          )}
        </Box>
      )}
    </Box>
  )
}

export function Panel({ sections, t, title }: PanelProps) {
  return (
    <Box borderColor={t.color.border} borderStyle="round" flexDirection="column" paddingX={2} paddingY={1}>
      <Box justifyContent="center" marginBottom={1}>
        <Text bold color={t.color.primary}>
          {title}
        </Text>
      </Box>

      {sections.map((sec, si) => (
        <Box flexDirection="column" key={si} marginTop={si > 0 ? 1 : 0}>
          {sec.title && (
            <Text bold color={t.color.accent}>
              {sec.title}
            </Text>
          )}

          {sec.rows?.map(([k, v], ri) => (
            <Text key={ri} wrap="truncate">
              <Text color={t.color.muted}>{k.padEnd(20)}</Text>
              <Text color={t.color.text}>{v}</Text>
            </Text>
          ))}

          {sec.items?.map((item, ii) => (
            <Text color={t.color.text} key={ii} wrap="truncate">
              {item}
            </Text>
          ))}

          {sec.text && <Text color={t.color.muted}>{sec.text}</Text>}
        </Box>
      ))}
    </Box>
  )
}

interface PanelProps {
  sections: PanelSection[]
  t: Theme
  title: string
}

interface SessionPanelProps {
  info: SessionInfo
  maxWidth?: number
  sid?: string | null
  t: Theme
}
