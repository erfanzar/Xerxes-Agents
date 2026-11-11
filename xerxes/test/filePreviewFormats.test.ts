// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'
import { deflateRawSync } from 'node:zlib'

import { parseDelimited, readDelimited, sniffDelimiter } from '../src/daemon/previews/delimited.js'
import { clockOf, formatCellNumber, generalNumber } from '../src/daemon/previews/numberFormat.js'
import { columnName, columnOf, readWorkbook, WorkbookFormatError } from '../src/daemon/previews/spreadsheet.js'
import { DocumentFormatError, readWordDocument } from '../src/daemon/previews/wordDocument.js'
import { parseXmlTree, XmlTooComplexError } from '../src/daemon/previews/xmlTree.js'
import { OfficeZipEntryTooLargeError, UnsupportedOfficeZipError } from '../src/skills/powerpoint/errors.js'
import { createOfficeZip, openOfficeZip, readOfficeZip } from '../src/skills/powerpoint/zip.js'
import { documentFixture, workbookFixture } from './fixtures/officePreview.js'

const bytes = (text: string): Uint8Array => new TextEncoder().encode(text)

describe('CSV and TSV', () => {
  test('quoted fields keep delimiters, doubled quotes and line breaks; any line ending ends a record', () => {
    const table = parseDelimited('id,name,note\r\n1,"Smith, J","said ""hi"""\n2,"two\nlines",x\r3,plain,"" \n', ',')
    expect(table.header).toEqual(['id', 'name', 'note'])
    expect(table.rows).toEqual([['1', 'Smith, J', 'said "hi"'], ['2', 'two\nlines', 'x'], ['3', 'plain', ' ']])
    expect(table).toMatchObject({ totalRows: 3, totalColumns: 3, truncated: false })
  })

  test('blank lines are not records, ragged rows are padded, a last line needs no line break', () => {
    const table = parseDelimited('a,b,c\n\n1\n2,3,4,5', ',')
    expect(table.header).toEqual(['a', 'b', 'c', ''])
    expect(table.rows).toEqual([['1', '', '', ''], ['2', '3', '4', '5']])
    expect(table.totalColumns).toBe(4)
  })

  test('an unterminated quote keeps the rest of the file as one field instead of failing', () => {
    expect(parseDelimited('a,b\n1,"open\n2,3', ',').rows).toEqual([['1', 'open\n2,3']])
  })

  test('a quote inside an unquoted field is a literal character', () => {
    expect(parseDelimited('size\n5" screen\n', ',').rows).toEqual([['5" screen']])
  })

  test('the delimiter is sniffed: European Excel writes semicolons, quoted commas do not count', () => {
    expect(sniffDelimiter('name;amount\n"Smith, J";10,5\nLee;7\n')).toBe(';')
    expect(sniffDelimiter('a|b|c\n1|2|3\n')).toBe('|')
    expect(sniffDelimiter('single column\nvalue\n')).toBe(',')
    // .tsv always splits on tabs, even when commas outnumber them.
    expect(readDelimited(bytes('a,1\tb,2\n'), 'data.tsv').header).toEqual(['a,1', 'b,2'])
  })

  test('UTF-8 BOM, UTF-16 with a BOM and Windows-1252 bytes decode as spreadsheet apps write them', () => {
    expect(readDelimited(new Uint8Array([0xef, 0xbb, 0xbf, ...bytes('name\nJosé\n')]), 'a.csv')).toMatchObject({ encoding: 'utf-8', header: ['name'], rows: [['José']] })
    const utf16 = new Uint8Array([0xff, 0xfe, ...Array.from('k\tv\n1\t2\n').flatMap(char => [char.charCodeAt(0), 0])])
    expect(readDelimited(utf16, 'export.txt.tsv')).toMatchObject({ encoding: 'utf-16le', header: ['k', 'v'], rows: [['1', '2']] })
    expect(readDelimited(new Uint8Array([...bytes('city\nM'), 0xfc, ...bytes('nchen\n')]), 'a.csv')).toMatchObject({ encoding: 'windows-1252', rows: [['München']] })
  })

  test('every record is counted, but only the first rows, columns and characters are kept', () => {
    const limits = { maxRows: 2, maxColumns: 2, maxCellChars: 5, maxChars: 1000 }
    const table = parseDelimited(['h1,h2,h3', 'abcdefgh,2,3', 'x,y,z', 'p,q,r', 'last,row,!'].join('\n'), ',', limits)
    expect(table.header).toEqual(['h1', 'h2'])
    expect(table.rows).toEqual([['abcd…', '2'], ['x', 'y']])
    expect(table).toMatchObject({ totalRows: 4, totalColumns: 3, truncated: true })
    // The character budget stops keeping rows before the row cap does.
    const budget = parseDelimited('h\n' + 'x'.repeat(40) + '\n' + 'y'.repeat(40) + '\n', ',', { maxRows: 10, maxColumns: 10, maxCellChars: 100, maxChars: 50 })
    expect(budget.rows.length).toBe(1)
    expect(budget).toMatchObject({ totalRows: 2, truncated: true })
  })

  test('an empty file is an empty table', () => {
    expect(parseDelimited('', ',')).toMatchObject({ header: [], rows: [], totalRows: 0, totalColumns: 0, truncated: false })
  })
})

describe('Excel number formats', () => {
  test('dates and times read as ISO dates and clock times, never serial numbers', () => {
    expect(formatCellNumber(45231, 'yyyy-mm-dd', false)).toBe('2023-11-01')
    expect(formatCellNumber(45231.75, 'm/d/yy h:mm', false)).toBe('2023-11-01 18:00')
    expect(formatCellNumber(0.5, 'h:mm:ss', false)).toBe('12:00:00')
    expect(formatCellNumber(1.25, '[h]:mm:ss', false)).toBe('30:00:00')
    // The 1900 system counts Excel's fictional 1900-02-29; the 1904 system does not.
    expect(formatCellNumber(61, 'yyyy-mm-dd', false)).toBe('1900-03-01')
    expect(formatCellNumber(59, 'yyyy-mm-dd', false)).toBe('1900-02-28')
    expect(formatCellNumber(0, 'yyyy-mm-dd', true)).toBe('1904-01-01')
    // Quoted text and colors are not date letters.
    expect(clockOf('"days: "0')).toBeUndefined()
    expect(clockOf('[Red]0.00')).toBeUndefined()
    expect(clockOf('dd/mm/yyyy')).toMatchObject({ date: true, time: false })
  })

  test('percentages, fixed decimals, grouping, currency and negative sections', () => {
    expect(formatCellNumber(0.125, '0.00%', false)).toBe('12.50%')
    expect(formatCellNumber(0.5, '0%', false)).toBe('50%')
    expect(formatCellNumber(1234567.891, '#,##0.00', false)).toBe('1,234,567.89')
    expect(formatCellNumber(1234.5, '"$"#,##0.00', false)).toBe('$1,234.50')
    expect(formatCellNumber(1234.5, '[$€-407]#,##0', false)).toBe('€1,235')
    expect(formatCellNumber(-42, '#,##0.00;(#,##0.00)', false)).toBe('(42.00)')
    expect(formatCellNumber(-42, '0.0', false)).toBe('-42.0')
    expect(formatCellNumber(0, '0.00;-0.00;"zero"', false)).toBe('zero')
    expect(formatCellNumber(1234567, '#,##0,', false)).toBe('1,235')
    expect(formatCellNumber(7, '0 "kg"', false)).toBe('7 kg')
  })

  test('formats this does not cover fall back to the plain number, General drops float noise', () => {
    expect(formatCellNumber(0.000123, '0.00E+00', false)).toBe('0.000123')
    expect(formatCellNumber(1.5, '# ?/?', false)).toBe('1.5')
    expect(formatCellNumber(3, 'General', false)).toBe('3')
    expect(generalNumber(0.1 + 0.2)).toBe('0.3')
    expect(generalNumber(123456789012)).toBe('123456789012')
  })
})

describe('XLSX workbooks', () => {
  test('cells show as Excel displays them, from A1, with sheet tabs listed', async () => {
    const workbook = await readWorkbook(openOfficeZip(workbookFixture()))
    expect(workbook.sheets).toEqual([{ name: 'Data', hidden: false }, { name: 'Secret & Co', hidden: true }, { name: 'Chart', hidden: false }])
    expect(workbook.sheet).toBe(0)
    expect(workbook.rows).toEqual([
      ['Item', 'Date', 'Amount', 'Rate', 'Paid', ''],
      ['Widget A', '2023-11-01', '$1,234.50', '12.50%', 'TRUE', ''],
      ['Gadget', '2023-11-02', '(42.00)', '#DIV/0!', 'FALSE', ''],
      ['', '', '', '', '', ''],
      ['=SUM(C2:C3)', '', '1192.5', '', '', 'note & more'],
    ])
    expect(workbook).toMatchObject({ totalRows: 5, totalColumns: 6, truncated: false })
  })

  test('another sheet by index; past the last sheet shows the last; a chart sheet says so', async () => {
    const zip = openOfficeZip(workbookFixture())
    expect(await readWorkbook(zip, 1)).toMatchObject({ sheet: 1, rows: [['', ''], ['', 'hidden value']], totalRows: 2, totalColumns: 2 })
    expect(await readWorkbook(zip, 99)).toMatchObject({ sheet: 2, rows: [], notice: expect.stringContaining('chart') })
  })

  test('prefixed SpreadsheetML reads the same', async () => {
    expect((await readWorkbook(openOfficeZip(workbookFixture({ prefixed: true })))).rows[1]).toEqual(['Widget A', '2023-11-01', '$1,234.50', '12.50%', 'TRUE', ''])
  })

  test('large sheets keep the first rows and still report their full extent', async () => {
    const workbook = await readWorkbook(openOfficeZip(workbookFixture({ extraRows: 30 })), 0, { maxRows: 10, maxColumns: 3, maxCellChars: 100, maxChars: 10_000 })
    expect(workbook.rows.length).toBe(10)
    expect(workbook.rows[0]).toEqual(['Item', 'Date', 'Amount'])
    expect(workbook).toMatchObject({ totalRows: 35, totalColumns: 6, truncated: true })
  })

  test('cell references and column names round-trip', () => {
    expect(columnOf('A1')).toBe(0)
    expect(columnOf('$AB$12')).toBe(27)
    expect(columnName(0)).toBe('A')
    expect(columnName(27)).toBe('AB')
    expect(columnName(16383)).toBe('XFD')
  })

  test('an archive that is not a workbook is a typed error', async () => {
    const notWorkbook = createOfficeZip([{ name: 'word/document.xml', data: bytes('<w:document/>') }])
    await expect(readWorkbook(openOfficeZip(notWorkbook))).rejects.toThrow(WorkbookFormatError)
  })
})

describe('DOCX documents', () => {
  test('headings, runs, lists with Word markers, tables and line breaks come through; hidden and deleted text do not', () => {
    const { blocks, truncated } = readWordDocument(openOfficeZip(documentFixture()))
    expect(truncated).toBe(false)
    expect(blocks).toEqual([
      { type: 'heading', level: 1, text: 'Quarterly Review' },
      // A localized style id still reads as "heading 1" by its name.
      { type: 'heading', level: 1, text: 'Summary' },
      { type: 'paragraph', runs: [{ text: 'Revenue grew ' }, { text: '12%', bold: true }, { text: ' on ' }, { text: 'strong', italic: true }, { text: ' demand.' }] },
      { type: 'paragraph', runs: [{ text: 'See the dashboard' }] },
      { type: 'list_item', ordered: false, level: 0, marker: '•', runs: [{ text: 'First point' }] },
      { type: 'list_item', ordered: false, level: 1, marker: '◦', runs: [{ text: 'Nested point' }] },
      { type: 'list_item', ordered: false, level: 0, marker: '•', runs: [{ text: 'Second point' }] },
      { type: 'list_item', ordered: true, level: 0, marker: '1.', runs: [{ text: 'Step one' }] },
      { type: 'list_item', ordered: true, level: 1, marker: '1.a)', runs: [{ text: 'Step one a' }] },
      { type: 'list_item', ordered: true, level: 1, marker: '1.b)', runs: [{ text: 'Step one b' }] },
      { type: 'list_item', ordered: true, level: 0, marker: '2.', runs: [{ text: 'Step two' }] },
      // A deeper level restarts under a new parent item.
      { type: 'list_item', ordered: true, level: 1, marker: '2.a)', runs: [{ text: 'Step two a' }] },
      { type: 'heading', level: 2, text: 'Details' },
      { type: 'paragraph', runs: [{ text: 'Inside a content control' }] },
      { type: 'table', rows: [['Region', 'Revenue'], ['EMEA', '1,200'], ['Total', '']] },
      { type: 'paragraph', runs: [{ text: 'Line one\nLine two\ttabbed' }] },
    ])
  })

  test('long documents stop at the block budget and say so', () => {
    const preview = readWordDocument(openOfficeZip(documentFixture()), { maxBlocks: 3, maxChars: 1_000_000, maxTableRows: 10, maxTableColumns: 10 })
    expect(preview.blocks.length).toBe(3)
    expect(preview.truncated).toBe(true)
  })

  test('an archive without a document part is a typed error', () => {
    expect(() => readWordDocument(openOfficeZip(workbookFixture()))).toThrow(DocumentFormatError)
  })
})

describe('bounded reading', () => {
  test('entries are inflated only when read, and never past the caller’s limit', () => {
    const big = new Uint8Array(4 * 1024 * 1024)
    const zip = openOfficeZip(createOfficeZip([{ name: 'xl/worksheets/sheet1.xml', data: big }, { name: 'small.xml', data: bytes('<a/>') }]))
    expect(zip.names).toEqual(['small.xml', 'xl/worksheets/sheet1.xml'])
    expect(new TextDecoder().decode(zip.read('small.xml', 10))).toBe('<a/>')
    expect(() => zip.read('xl/worksheets/sheet1.xml', 1024)).toThrow(OfficeZipEntryTooLargeError)
    expect(zip.read('missing.xml')).toBeUndefined()
    // Part names compare case-insensitively.
    expect(zip.read('XL/Worksheets/Sheet1.xml', big.length)?.length).toBe(big.length)
  })

  test('an entry that lies about its size is still stopped by the inflater', () => {
    const honest = createOfficeZip([{ name: 'a.xml', data: new Uint8Array(2 * 1024 * 1024) }])
    // Rewrite the declared uncompressed size (central directory offset 24) to 10 bytes.
    const view = new DataView(honest.buffer, honest.byteOffset, honest.byteLength)
    const central = honest.findLastIndex((_, index) => index + 4 <= honest.length && view.getUint32(index, true) === 0x02014b50)
    view.setUint32(central + 24, 10, true)
    expect(() => openOfficeZip(honest).read('a.xml', 1024)).toThrow(OfficeZipEntryTooLargeError)
  })

  test('the eager reader other tools use is unchanged', () => {
    const files = readOfficeZip(createOfficeZip([{ name: 'a.xml', data: bytes('<a/>') }, { name: 'b/c.xml', data: deflateRawSync(bytes('x')) }]))
    expect([...files.keys()]).toEqual(['a.xml', 'b/c.xml'])
    expect(() => readOfficeZip(bytes('not a zip'))).toThrow(UnsupportedOfficeZipError)
  })

  test('XML: no entity expansion, DTDs skipped, prefixes dropped, damaged nesting tolerated, element budget enforced', () => {
    const tree = parseXmlTree('<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;">]><w:doc xmlns:w="urn:w" w:val="1 &amp; 2"><w:t>&b; &lt;ok&gt; &#x41;</w:t><!-- note --><![CDATA[<raw>]]><w:p><w:r></w:p></w:doc>')
    const doc = tree.children[0]
    if (!doc || typeof doc === 'string') throw new Error('expected the root element')
    expect(doc).toMatchObject({ name: 'doc', attributes: { val: '1 & 2' } })
    expect(doc.children[0]).toMatchObject({ name: 't', children: ['&b; <ok> A'] })
    expect(doc.children[1]).toBe('<raw>')
    expect(() => parseXmlTree('<a><b/><b/><b/></a>', 3)).toThrow(XmlTooComplexError)
  })
})
