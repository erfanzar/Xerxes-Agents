// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";

import {
  OCR_DOCUMENTS_CLI_USAGE,
  runOcrDocumentsCli,
  type OcrDocumentsCliDependencies,
} from "../src/skills/ocrDocuments/cli.js";
import type {
  OcrDocumentFilesystemPort,
  PdfTextConversionPort,
} from "../src/skills/ocrDocuments/types.js";

test("OCR document CLI exposes help and refuses ambient PDF engine discovery", async () => {
  const output: string[] = [];
  expect(
    await runOcrDocumentsCli([], {
      writeLine: (line) => {
        output.push(line);
      },
    }),
  ).toBe(0);
  expect(output).toEqual([OCR_DOCUMENTS_CLI_USAGE]);

  output.length = 0;
  expect(
    await runOcrDocumentsCli(["pymupdf", "report.pdf"], {
      writeLine: (line) => {
        output.push(line);
      },
    }),
  ).toBe(2);
  expect(output[0]).toContain("caller-provided Bun-native pdfConverter");
});

test("OCR document CLI maps every former PyMuPDF helper mode through an explicit native adapter", async () => {
  const output: string[] = [];
  const filesystem = createFilesystem();
  let adapterPath = "";
  const dependencies: OcrDocumentsCliDependencies = {
    loadAdapter: (path) => {
      adapterPath = path;
      return {
        filesystem: filesystem.port,
        pdfConverter: createPdfConverter(),
      };
    },
    writeLine: (line) => {
      output.push(line);
    },
  };
  const adapter = ["--adapter", "./fixture.ocr-documents.ts"];

  expect(
    await runOcrDocumentsCli(
      [...adapter, "pymupdf", "report.pdf", "--pages", "0"],
      dependencies,
    ),
  ).toBe(0);
  expect(output.at(-1)).toBe("\n--- Page 1/2 ---\n\nRevenue rose.");

  expect(
    await runOcrDocumentsCli(
      [...adapter, "pymupdf", "report.pdf", "--markdown", "--pages", "1"],
      dependencies,
    ),
  ).toBe(0);
  expect(output.at(-1)).toBe("# Page two");

  expect(
    await runOcrDocumentsCli(
      [...adapter, "pymupdf", "report.pdf", "--tables"],
      dependencies,
    ),
  ).toBe(0);
  expect(output.at(-1)).toContain("--- Page 1, Table 1 ---");

  expect(
    await runOcrDocumentsCli(
      [...adapter, "pymupdf", "report.pdf", "--images", "pdf-images"],
      dependencies,
    ),
  ).toBe(0);
  expect(output.at(-1)).toBe("Extracted 2 images to pdf-images/");
  expect([...filesystem.files.keys()]).toEqual([
    "pdf-images/page1_img1.png",
    "pdf-images/page2_img1.png",
  ]);

  expect(
    await runOcrDocumentsCli(
      [...adapter, "pymupdf", "report.pdf", "--metadata"],
      dependencies,
    ),
  ).toBe(0);
  expect(JSON.parse(output.at(-1) ?? "{}")).toMatchObject({
    pages: 2,
    title: "Quarterly report",
  });
  expect(adapterPath).toBe("./fixture.ocr-documents.ts");
});

test("OCR document CLI maps marker checks and conversion flags through explicit native ports", async () => {
  const diagnostics: string[] = [];
  const output: string[] = [];
  const filesystem = createFilesystem();
  const requests: unknown[] = [];
  const dependencies: OcrDocumentsCliDependencies = {
    diskSpace: { freeBytes: async () => 5 * 1024 ** 3 },
    filesystem: filesystem.port,
    markerConverter: {
      async convert(request) {
        requests.push(request);
        return {
          images: { "scan.png": new Uint8Array([7, 8]) },
          markdown: "# Scan\n\nRecovered text.",
          metadata: { confidence: 0.98, language: "en" },
        };
      },
    },
    writeDiagnostic: (line) => {
      diagnostics.push(line);
    },
    writeLine: (line) => {
      output.push(line);
    },
  };

  expect(await runOcrDocumentsCli(["marker", "--check"], dependencies)).toBe(0);
  expect(output.at(-1)).toBe("✓ 5.0GB free — sufficient for marker-pdf");

  expect(
    await runOcrDocumentsCli(
      [
        "marker",
        "scanned.pdf",
        "--json",
        "--output_dir",
        "ocr-images",
        "--use_llm",
      ],
      dependencies,
    ),
  ).toBe(0);
  expect(requests).toEqual([{ path: "scanned.pdf", useLlm: true }]);
  expect(JSON.parse(output.at(-1) ?? "{}")).toEqual({
    markdown: "# Scan\n\nRecovered text.",
    metadata: { confidence: 0.98, language: "en" },
  });
  expect(diagnostics).toEqual(["Saved 1 image(s) to ocr-images/"]);
  expect([...filesystem.files.entries()]).toContainEqual([
    "ocr-images/scan.png",
    [7, 8],
  ]);

  const insufficient: string[] = [];
  expect(
    await runOcrDocumentsCli(["marker", "--check", "--path", "/workspace"], {
      diskSpace: { freeBytes: async () => 5 * 1024 ** 3 - 1 },
      writeLine: (line) => {
        insufficient.push(line);
      },
    }),
  ).toBe(1);
  expect(insufficient).toEqual([
    expect.stringContaining("Only 5.0GB free. marker-pdf needs ~5GB"),
  ]);
});

function createPdfConverter(): PdfTextConversionPort {
  return {
    async open() {
      return {
        metadata: {
          creator: "PDF.js",
          format: "PDF 1.7",
          title: "Quarterly report",
        },
        pageCount: 2,
        async page(index) {
          if (index === 0) {
            return {
              async extractImages() {
                return [{ bytes: new Uint8Array([1]) }];
              },
              async extractTables() {
                return [
                  {
                    headers: ["Item", "Value"],
                    rows: [["Revenue", 42]],
                  },
                ];
              },
              async extractText() {
                return "Revenue rose.";
              },
            };
          }
          return {
            async extractImages() {
              return [{ bytes: new Uint8Array([2]) }];
            },
            async extractTables() {
              return [];
            },
            async extractText() {
              return "Expenses fell.";
            },
          };
        },
      };
    },
    async toMarkdown(_path, pages) {
      expect(pages).toEqual([1]);
      return "# Page two";
    },
  };
}

function createFilesystem(): {
  readonly files: Map<string, number[]>;
  readonly port: OcrDocumentFilesystemPort;
} {
  const files = new Map<string, number[]>();
  return {
    files,
    port: {
      async ensureDirectory() {},
      join(directory, name) {
        return `${directory}/${name}`;
      },
      async writeFile(path, bytes) {
        files.set(path, [...bytes]);
      },
    },
  };
}
