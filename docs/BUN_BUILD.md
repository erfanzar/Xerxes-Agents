# Native Bun documentation build

Build the static documentation site with:

```sh
bun run --cwd src/typescript docs:build
```

The command reads `bun-docs.json`, writes deterministic HTML into `docs/_bun/`, and generates
a native TypeScript API reference in `docs/_bun/typescript-api/`. It uses Bun and TypeScript only.

## Source support

Markdown is rendered as static HTML with headings, paragraphs, lists, block quotes, fenced code,
and ordinary links. The API reference is generated directly from the native TypeScript runtime.

The generated `documentation-report.html` and `build-report.json` record build notices and the
native API inventory. Markdown is the only handwritten documentation input format.
