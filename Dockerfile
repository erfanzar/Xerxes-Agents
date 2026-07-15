# syntax=docker/dockerfile:1
# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0.

FROM oven/bun:1.3.12 AS build

WORKDIR /app

COPY package.json bun.lock ./
COPY xerxes/package.json xerxes/package.json
RUN bun install --frozen-lockfile

COPY . ./
RUN bun run build

FROM oven/bun:1.3.12 AS runtime

ARG XERXES_UID=1000

# Reuse the unprivileged account supplied by the official Bun image. Matching
# its UID to the Linux checkout owner keeps a bind-mounted /workspace writable
# without running the daemon as root.
RUN groupmod --new-name xerxes bun \
    && usermod --login xerxes --uid "$XERXES_UID" --gid xerxes \
      --home-dir /home/xerxes --move-home bun

WORKDIR /app
ENV HOME=/home/xerxes \
    XERXES_ENV=production \
    XERXES_HOME=/home/xerxes/.xerxes

COPY packaging/runtime/package.json packaging/runtime/bun.lock ./
RUN bun install --frozen-lockfile --production

COPY --from=build --chown=xerxes:xerxes /app/xerxes/dist/cli.js ./xerxes/dist/cli.js
COPY --from=build --chown=xerxes:xerxes /app/xerxes/dist/default ./xerxes/dist/default
COPY --from=build --chown=xerxes:xerxes /app/xerxes/dist/skills ./xerxes/dist/skills
COPY --from=build --chown=xerxes:xerxes /app/xerxes/dist/ui/entry.js ./xerxes/dist/ui/entry.js
RUN mkdir -p "$XERXES_HOME" /workspace \
    && chown -R xerxes:xerxes /app /home/xerxes /workspace

USER xerxes

EXPOSE 11996

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD ["bun", "/app/xerxes/dist/cli.js", "--version"]

ENTRYPOINT ["bun", "/app/xerxes/dist/cli.js"]
CMD ["daemon"]
