// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  requireSkillText,
  skillFetchJson,
  skillJsonArray,
  skillJsonObject,
  type SkillFetch,
  type SkillJsonObject,
} from './http.js'

/** Public OSM endpoints used by the find-nearby bundled skill. */
export const OVERPASS_URLS = [
  'https://overpass-api.de/api/interpreter',
  'https://overpass.kumi.systems/api/interpreter',
] as const
export const NOMINATIM_URL = 'https://nominatim.openstreetmap.org/search'
export const NEARBY_USER_AGENT = 'Xerxes/1.0 (find-nearby skill)'

export interface GeographicPoint {
  readonly lat: number
  readonly lon: number
}

export interface NearbyPlace extends GeographicPoint {
  readonly address?: string
  readonly cuisine?: string
  readonly directionsUrl: string
  readonly distanceMeters: number
  readonly hours?: string
  readonly mapsUrl: string
  readonly name: string
  readonly phone?: string
  readonly type: string
  readonly website?: string
}

export interface FindNearbyOptions {
  readonly limit?: number
  readonly radius?: number
  readonly signal?: AbortSignal
  readonly types: readonly string[]
}

export interface FindNearbyClientOptions {
  readonly fetchImplementation?: SkillFetch
  readonly geocodeUrl?: string
  readonly overpassUrls?: readonly string[]
}

/** Calculate a great-circle distance in meters using the Haversine formula. */
export function haversineMeters(from: GeographicPoint, to: GeographicPoint): number {
  assertCoordinates(from, 'from')
  assertCoordinates(to, 'to')
  const radius = 6_371_000
  const fromLatitude = degreesToRadians(from.lat)
  const toLatitude = degreesToRadians(to.lat)
  const latitudeDelta = degreesToRadians(to.lat - from.lat)
  const longitudeDelta = degreesToRadians(to.lon - from.lon)
  const factor = Math.sin(latitudeDelta / 2) ** 2
    + Math.cos(fromLatitude) * Math.cos(toLatitude) * Math.sin(longitudeDelta / 2) ** 2
  return radius * 2 * Math.atan2(Math.sqrt(factor), Math.sqrt(1 - factor))
}

/** Build the Overpass query used to search amenities around one coordinate. */
export function overpassNearbyQuery(origin: GeographicPoint, types: readonly string[], radius = 1_500): string {
  assertCoordinates(origin, 'origin')
  const normalizedTypes = normalizeTypes(types)
  assertRadius(radius)
  const filters = normalizedTypes
    .map(type => `nwr["amenity"="${overpassString(type)}"](around:${radius},${origin.lat},${origin.lon});`)
    .join('')
  return `[out:json][timeout:15];(${filters});out center tags;`
}

/** Render nearby-place data in the original skill's readable text layout. */
export function formatNearbyPlaces(places: readonly NearbyPlace[], types: readonly string[], radius: number): string {
  if (!places.length) return `No ${normalizeTypes(types).join('/')} found within ${radius}m`
  const lines = [`Found ${places.length} places within ${radius}m:`, '']
  for (const [index, place] of places.entries()) {
    const distance = place.distanceMeters < 1_000
      ? `${place.distanceMeters}m`
      : `${(place.distanceMeters / 1_000).toFixed(1)}km`
    lines.push(`  ${index + 1}. ${place.name} (${place.type}) — ${distance}`)
    if (place.cuisine) lines.push(`     Cuisine: ${place.cuisine}`)
    if (place.hours) lines.push(`     Hours: ${place.hours}`)
    if (place.address) lines.push(`     Address: ${place.address}`)
    lines.push(`     Map: ${place.mapsUrl}`, '')
  }
  return lines.join('\n').trimEnd()
}

/** Native OpenStreetMap client with an injected fetch seam and Overpass fallback endpoints. */
export class FindNearbyClient {
  private readonly fetchImplementation: SkillFetch
  private readonly geocodeUrl: string
  private readonly overpassUrls: readonly string[]

  constructor(options: FindNearbyClientOptions = {}) {
    this.fetchImplementation = options.fetchImplementation ?? fetch
    this.geocodeUrl = requireSkillText(options.geocodeUrl ?? NOMINATIM_URL, 'geocodeUrl')
    this.overpassUrls = options.overpassUrls ?? OVERPASS_URLS
    if (!this.overpassUrls.length) throw new TypeError('overpassUrls must contain at least one endpoint')
  }

  /** Resolve an address, city, or postal code through Nominatim. */
  async geocode(query: string, signal?: AbortSignal): Promise<GeographicPoint> {
    const url = new URL(this.geocodeUrl)
    url.searchParams.set('q', requireSkillText(query, 'query'))
    url.searchParams.set('format', 'json')
    url.searchParams.set('limit', '1')
    const data = await skillFetchJson(this.fetchImplementation, url, this.requestInit(signal))
    const entries = skillJsonArray(data, 'Nominatim geocode response')
    const first = entries[0]
    if (first === undefined) throw new Error(`Could not geocode '${query}'. Try a more specific address.`)
    const result = skillJsonObject(first, 'Nominatim geocode result')
    const point = { lat: coordinate(result.lat, 'Nominatim latitude'), lon: coordinate(result.lon, 'Nominatim longitude') }
    assertCoordinates(point, 'Nominatim result')
    return point
  }

  /** Return nearby named amenities, sorted by distance and limited after extraction. */
  async findNearby(origin: GeographicPoint, options: FindNearbyOptions): Promise<readonly NearbyPlace[]> {
    assertCoordinates(origin, 'origin')
    const types = normalizeTypes(options.types)
    const radius = options.radius ?? 1_500
    const limit = options.limit ?? 15
    assertRadius(radius)
    if (!Number.isSafeInteger(limit) || limit < 0) throw new RangeError('limit must be a non-negative safe integer')
    const query = overpassNearbyQuery(origin, types, radius)
    const body = `data=${encodeURIComponent(query)}`

    for (const endpoint of this.overpassUrls) {
      try {
        const data = await skillFetchJson(this.fetchImplementation, endpoint, {
          method: 'POST',
          body,
          headers: {
            Accept: 'application/json',
            'Content-Type': 'application/x-www-form-urlencoded',
            'User-Agent': NEARBY_USER_AGENT,
          },
          ...(options.signal === undefined ? {} : { signal: options.signal }),
        })
        const response = skillJsonObject(data, 'Overpass response')
        const places = extractNearbyPlaces(origin, response)
        return places.sort((left, right) => left.distanceMeters - right.distanceMeters).slice(0, limit)
      } catch (error) {
        if (options.signal?.aborted) throw error
        // Overpass publishes independent mirrors specifically for this fallback behavior.
      }
    }
    return []
  }

  private requestInit(signal?: AbortSignal): RequestInit {
    return {
      headers: { Accept: 'application/json', 'User-Agent': NEARBY_USER_AGENT },
      ...(signal === undefined ? {} : { signal }),
    }
  }
}

function extractNearbyPlaces(origin: GeographicPoint, response: SkillJsonObject): NearbyPlace[] {
  const elements = skillJsonArray(response.elements ?? [], 'Overpass elements')
  const places: NearbyPlace[] = []
  for (const [index, element] of elements.entries()) {
    const item = skillJsonObject(element, `Overpass element ${index}`)
    const tags = optionalRecord(item.tags)
    const name = textValue(tags?.name)
    if (!name) continue
    const center = optionalRecord(item.center)
    const lat = optionalCoordinate(item.lat) ?? optionalCoordinate(center?.lat)
    const lon = optionalCoordinate(item.lon) ?? optionalCoordinate(center?.lon)
    if (lat === undefined || lon === undefined) continue
    const target = { lat, lon }
    const cuisine = textValue(tags?.cuisine)
    const hours = textValue(tags?.opening_hours)
    const phone = textValue(tags?.phone)
    const website = textValue(tags?.website)
    const address = nearbyAddress(tags)
    places.push({
      ...(address ? { address } : {}),
      ...(cuisine ? { cuisine } : {}),
      directionsUrl: `https://www.google.com/maps/dir/?api=1&origin=${origin.lat},${origin.lon}&destination=${lat},${lon}`,
      distanceMeters: Math.round(haversineMeters(origin, target)),
      ...(hours ? { hours } : {}),
      lat,
      lon,
      mapsUrl: `https://www.google.com/maps/search/?api=1&query=${lat},${lon}`,
      name,
      ...(phone ? { phone } : {}),
      type: textValue(tags?.amenity),
      ...(website ? { website } : {}),
    })
  }
  return places
}

function normalizeTypes(types: readonly string[]): readonly string[] {
  const normalized = types.map(type => requireSkillText(type, 'amenity type'))
  if (!normalized.length) throw new TypeError('types must contain at least one amenity type')
  return normalized
}

function assertCoordinates(point: GeographicPoint, name: string): void {
  if (!Number.isFinite(point.lat) || point.lat < -90 || point.lat > 90) throw new RangeError(`${name}.lat must be between -90 and 90`)
  if (!Number.isFinite(point.lon) || point.lon < -180 || point.lon > 180) throw new RangeError(`${name}.lon must be between -180 and 180`)
}

function assertRadius(radius: number): void {
  if (!Number.isSafeInteger(radius) || radius <= 0) throw new RangeError('radius must be a positive safe integer')
}

function degreesToRadians(value: number): number {
  return value * Math.PI / 180
}

function coordinate(value: unknown, name: string): number {
  const parsed = optionalCoordinate(value)
  if (parsed === undefined) throw new TypeError(`${name} must be numeric`)
  return parsed
}

function optionalCoordinate(value: unknown): number | undefined {
  if (typeof value !== 'number' && typeof value !== 'string') return undefined
  if (typeof value === 'string' && !value.trim()) return undefined
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

function optionalRecord(value: unknown): SkillJsonObject | undefined {
  return value === null || typeof value !== 'object' || Array.isArray(value) ? undefined : value as SkillJsonObject
}

function textValue(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function nearbyAddress(tags: SkillJsonObject | undefined): string | undefined {
  if (tags === undefined || !textValue(tags['addr:street'])) return undefined
  return [textValue(tags['addr:housenumber']), textValue(tags['addr:street']), textValue(tags['addr:city'])]
    .filter(Boolean)
    .join(' ')
}

function overpassString(value: string): string {
  return value.replaceAll('\\', '\\\\').replaceAll('"', '\\"')
}
