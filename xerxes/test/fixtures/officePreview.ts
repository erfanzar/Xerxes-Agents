// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Small but realistic XLSX and DOCX packages for preview tests, written the
 * way Excel and Word write them: relationships, shared strings, cell styles
 * with number formats, localized style ids, numbering definitions.
 */

import { createOfficeZip } from '../../src/skills/powerpoint/zip.js'

const encoder = new TextEncoder()
const part = (name: string, xml: string) => ({ name, data: encoder.encode(xml) })
const XML = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
const REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
const SHEET_NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
const WORD_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'

const rootRels = (target: string) => part('_rels/.rels', `${XML}<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="${REL}/officeDocument" Target="${target}"/></Relationships>`)

export interface WorkbookFixtureOptions {
  /** Extra rows for the first sheet, after the fixture's own. */
  readonly extraRows?: number
  /** Write the sheet with an `x:` prefix, as some generators do. */
  readonly prefixed?: boolean
}

/** Three sheets: Data (strings, numbers under formats, dates, booleans, errors, a formula), a hidden sheet and a chart sheet. */
export function workbookFixture(options: WorkbookFixtureOptions = {}): Uint8Array {
  const p = options.prefixed ? 'x:' : ''
  const ns = options.prefixed ? `xmlns:x="${SHEET_NS}"` : `xmlns="${SHEET_NS}"`
  const extra = Array.from({ length: options.extraRows ?? 0 }, (_, index) => `<${p}row r="${index + 6}"><${p}c r="A${index + 6}"><${p}v>${index}</${p}v></${p}c></${p}row>`).join('')
  const sheet1 = `${XML}<${p}worksheet ${ns} xmlns:r="${REL}"><${p}dimension ref="A1:F5"/><${p}sheetData>`
    + `<${p}row r="1"><${p}c r="A1" t="s"><${p}v>0</${p}v></${p}c><${p}c r="B1" t="s"><${p}v>1</${p}v></${p}c><${p}c r="C1" t="s"><${p}v>2</${p}v></${p}c><${p}c r="D1" t="s"><${p}v>3</${p}v></${p}c><${p}c r="E1" t="inlineStr"><${p}is><${p}t>Paid</${p}t></${p}is></${p}c></${p}row>`
    + `<${p}row r="2"><${p}c r="A2" t="s"><${p}v>4</${p}v></${p}c><${p}c r="B2" s="1"><${p}v>45231</${p}v></${p}c><${p}c r="C2" s="2"><${p}v>1234.5</${p}v></${p}c><${p}c r="D2" s="3"><${p}v>0.125</${p}v></${p}c><${p}c r="E2" t="b"><${p}v>1</${p}v></${p}c></${p}row>`
    + `<${p}row r="3"><${p}c r="A3" t="s"><${p}v>5</${p}v></${p}c><${p}c r="B3" s="1"><${p}v>45232.5</${p}v></${p}c><${p}c r="C3" s="4"><${p}v>-42</${p}v></${p}c><${p}c r="D3" t="e"><${p}v>#DIV/0!</${p}v></${p}c><${p}c r="E3" t="b"><${p}v>0</${p}v></${p}c></${p}row>`
    + `<${p}row r="5"><${p}c r="A5"><${p}f>SUM(C2:C3)</${p}f></${p}c><${p}c r="C5"><${p}f>SUM(C2:C3)</${p}f><${p}v>1192.5</${p}v></${p}c><${p}c r="F5" t="str"><${p}v>note &amp; more</${p}v></${p}c></${p}row>`
    + extra
    + `</${p}sheetData></${p}worksheet>`
  return createOfficeZip([
    part('[Content_Types].xml', `${XML}<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>`),
    rootRels('xl/workbook.xml'),
    part('xl/workbook.xml', `${XML}<workbook xmlns="${SHEET_NS}" xmlns:r="${REL}"><workbookPr/><sheets><sheet name="Data" sheetId="1" r:id="rId1"/><sheet name="Secret &amp; Co" sheetId="2" state="hidden" r:id="rId2"/><sheet name="Chart" sheetId="3" r:id="rId3"/></sheets></workbook>`),
    part('xl/_rels/workbook.xml.rels', `${XML}<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">`
      + `<Relationship Id="rId1" Type="${REL}/worksheet" Target="worksheets/sheet1.xml"/>`
      + `<Relationship Id="rId2" Type="${REL}/worksheet" Target="/xl/worksheets/sheet2.xml"/>`
      + `<Relationship Id="rId3" Type="${REL}/chartsheet" Target="chartsheets/sheet1.xml"/>`
      + `<Relationship Id="rId4" Type="${REL}/sharedStrings" Target="sharedStrings.xml"/>`
      + `<Relationship Id="rId5" Type="${REL}/styles" Target="styles.xml"/>`
      + `</Relationships>`),
    part('xl/sharedStrings.xml', `${XML}<sst xmlns="${SHEET_NS}" count="6" uniqueCount="6"><si><t>Item</t></si><si><t>Date</t></si><si><t>Amount</t></si><si><t>Rate</t></si>`
      + `<si><r><rPr><b/></rPr><t>Widget</t></r><r><t xml:space="preserve"> A</t></r></si><si><t>Gadget</t><rPh><t>GADGETTO</t></rPh></si></sst>`),
    part('xl/styles.xml', `${XML}<styleSheet xmlns="${SHEET_NS}"><numFmts count="1"><numFmt numFmtId="164" formatCode="&quot;$&quot;#,##0.00"/></numFmts>`
      + `<cellStyleXfs count="1"><xf numFmtId="0"/></cellStyleXfs>`
      + `<cellXfs count="5"><xf numFmtId="0"/><xf numFmtId="14"/><xf numFmtId="164"/><xf numFmtId="10"/><xf numFmtId="39"/></cellXfs></styleSheet>`),
    part('xl/worksheets/sheet1.xml', sheet1),
    part('xl/worksheets/sheet2.xml', `${XML}<worksheet xmlns="${SHEET_NS}"><sheetData><row r="2"><c r="B2" t="inlineStr"><is><t>hidden value</t></is></c></row></sheetData></worksheet>`),
    part('xl/chartsheets/sheet1.xml', `${XML}<chartsheet xmlns="${SHEET_NS}"/>`),
  ])
}

/** A report with a localized heading style, a title, bulleted and numbered lists, a table and text Word does not show. */
export function documentFixture(): Uint8Array {
  const w = (body: string) => `${XML}<w:document xmlns:w="${WORD_NS}" xmlns:r="${REL}"><w:body>${body}<w:sectPr/></w:body></w:document>`
  const p = (inner: string, props = '') => `<w:p>${props ? `<w:pPr>${props}</w:pPr>` : ''}${inner}</w:p>`
  const r = (text: string, props = '') => `<w:r>${props ? `<w:rPr>${props}</w:rPr>` : ''}<w:t xml:space="preserve">${text}</w:t></w:r>`
  const item = (text: string, numId: number, level = 0) => p(r(text), `<w:numPr><w:ilvl w:val="${level}"/><w:numId w:val="${numId}"/></w:numPr>`)
  const cell = (text: string, span = 1) => `<w:tc>${span > 1 ? `<w:tcPr><w:gridSpan w:val="${span}"/></w:tcPr>` : ''}${p(r(text))}</w:tc>`
  const body = [
    p(r('Quarterly Review'), '<w:pStyle w:val="Title"/>'),
    p(r('Summary'), '<w:pStyle w:val="berschrift1"/>'),
    p(r('Revenue grew ') + r('12%', '<w:b/>') + r(' on ') + r('strong', '<w:i/>') + r(' demand.') + '<w:del><w:r><w:delText>deleted words</w:delText></w:r></w:del>' + r(' secret', '<w:vanish/>')),
    p('<w:hyperlink r:id="rId9">' + r('See the dashboard') + '</w:hyperlink>'),
    item('First point', 1), item('Nested point', 1, 1), item('Second point', 1),
    item('Step one', 2), item('Step one a', 2, 1), item('Step one b', 2, 1), item('Step two', 2), item('Step two a', 2, 1),
    p(r('Details'), '<w:outlineLvl w:val="1"/>'),
    '<w:sdt><w:sdtPr/><w:sdtContent>' + p(r('Inside a content control')) + '</w:sdtContent></w:sdt>',
    `<w:tbl><w:tblPr/><w:tr>${cell('Region')}${cell('Revenue')}</w:tr><w:tr>${cell('EMEA')}${cell('1,200')}</w:tr><w:tr>${cell('Total', 2)}</w:tr></w:tbl>`,
    p(r('Line one') + '<w:r><w:br/><w:t>Line two</w:t><w:tab/><w:t>tabbed</w:t></w:r>'),
    p(''),
  ].join('')
  return createOfficeZip([
    part('[Content_Types].xml', `${XML}<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>`),
    rootRels('word/document.xml'),
    part('word/document.xml', w(body)),
    part('word/_rels/document.xml.rels', `${XML}<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">`
      + `<Relationship Id="rId1" Type="${REL}/styles" Target="styles.xml"/>`
      + `<Relationship Id="rId2" Type="${REL}/numbering" Target="numbering.xml"/>`
      + `<Relationship Id="rId9" Type="${REL}/hyperlink" Target="https://example.com" TargetMode="External"/>`
      + `</Relationships>`),
    part('word/styles.xml', `${XML}<w:styles xmlns:w="${WORD_NS}">`
      + `<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>`
      + `<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/></w:style>`
      + `<w:style w:type="paragraph" w:styleId="berschrift1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:outlineLvl w:val="0"/></w:pPr></w:style>`
      + `</w:styles>`),
    part('word/numbering.xml', `${XML}<w:numbering xmlns:w="${WORD_NS}">`
      + `<w:abstractNum w:abstractNumId="10"><w:lvl w:ilvl="0"><w:numFmt w:val="bullet"/><w:lvlText w:val=""/></w:lvl><w:lvl w:ilvl="1"><w:numFmt w:val="bullet"/><w:lvlText w:val="o"/></w:lvl></w:abstractNum>`
      + `<w:abstractNum w:abstractNumId="20"><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/></w:lvl><w:lvl w:ilvl="1"><w:start w:val="1"/><w:numFmt w:val="lowerLetter"/><w:lvlText w:val="%1.%2)"/></w:lvl></w:abstractNum>`
      + `<w:num w:numId="1"><w:abstractNumId w:val="10"/></w:num><w:num w:numId="2"><w:abstractNumId w:val="20"/></w:num>`
      + `</w:numbering>`),
  ])
}
