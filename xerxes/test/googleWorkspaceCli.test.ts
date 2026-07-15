// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";

import {
  GOOGLE_WORKSPACE_CLI_USAGE,
  runGoogleWorkspaceCli,
  type GoogleWorkspaceCliDependencies,
} from "../src/skills/googleWorkspace/cli.js";
import {
  GoogleWorkspaceClient,
  type GooglePendingAuthorization,
  type GoogleWorkspaceAuthorizationStorage,
  type GoogleWorkspaceEndpoints,
  type GoogleWorkspaceToken,
} from "../src/skills/googleWorkspace/index.js";

const ENDPOINTS: GoogleWorkspaceEndpoints = {
  calendar: "https://calendar.google.test/calendar/v3/",
  docs: "https://docs.google.test/v1/",
  drive: "https://drive.google.test/drive/v3/",
  gmail: "https://gmail.google.test/gmail/v1/",
  people: "https://people.google.test/v1/",
  sheets: "https://sheets.google.test/v4/",
};

class MemoryGoogleStorage implements GoogleWorkspaceAuthorizationStorage {
  pending: GooglePendingAuthorization | undefined;
  token: GoogleWorkspaceToken | undefined;

  async loadPendingAuthorization(): Promise<
    GooglePendingAuthorization | undefined
  > {
    return this.pending;
  }

  async loadToken(): Promise<GoogleWorkspaceToken | undefined> {
    return this.token;
  }

  async removePendingAuthorization(): Promise<void> {
    this.pending = undefined;
  }

  async removeToken(): Promise<void> {
    this.token = undefined;
  }

  async savePendingAuthorization(
    pending: GooglePendingAuthorization,
  ): Promise<void> {
    this.pending = pending;
  }

  async saveToken(token: GoogleWorkspaceToken): Promise<void> {
    this.token = token;
  }
}

function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    headers: { "Content-Type": "application/json" },
    status,
  });
}

function nativeCliClient(requests: URL[]): GoogleWorkspaceClient {
  return new GoogleWorkspaceClient({
    endpoints: ENDPOINTS,
    fetchImplementation: async (input, init) => {
      const url = new URL(String(input));
      requests.push(url);
      if (url.hostname === "gmail.google.test") {
        if (url.pathname.endsWith("/users/me/messages")) {
          return json({ messages: [] });
        }
        if (url.pathname.endsWith("/users/me/labels")) {
          return json({
            labels: [{ id: "INBOX", name: "Inbox", type: "system" }],
          });
        }
        if (url.pathname.endsWith("/modify")) {
          return json({ id: "message-1", labelIds: ["STARRED"] });
        }
        if (url.pathname.endsWith("/messages/send")) {
          return json({ id: "sent-1", threadId: "thread-1" });
        }
        if (url.searchParams.get("format") === "metadata") {
          return json({
            id: "message-1",
            labelIds: ["INBOX"],
            payload: {
              headers: [
                { name: "From", value: "sender@example.test" },
                { name: "Subject", value: "Status" },
                { name: "Message-ID", value: "<message-1@example.test>" },
              ],
            },
            snippet: "summary",
            threadId: "thread-1",
          });
        }
        return json({
          id: "message-1",
          labelIds: ["INBOX"],
          payload: { headers: [{ name: "Subject", value: "Status" }] },
          snippet: "summary",
          threadId: "thread-1",
        });
      }
      if (url.hostname === "calendar.google.test") {
        if (init?.method === "DELETE")
          return new Response(null, { status: 204 });
        if (init?.method === "POST") {
          return json({
            htmlLink: "https://calendar.google.test/event-1",
            id: "event-1",
            summary: "Planning",
          });
        }
        return json({ items: [] });
      }
      if (url.hostname === "drive.google.test") return json({ files: [] });
      if (url.hostname === "people.google.test")
        return json({ connections: [] });
      if (url.hostname === "sheets.google.test") {
        if (url.pathname.endsWith(":append")) {
          return json({ updates: { updatedCells: 2 } });
        }
        if (init?.method === "PUT") {
          return json({ updatedCells: 2, updatedRange: "Sheet1!A1:B1" });
        }
        return json({ values: [["one", 2]] });
      }
      if (url.hostname === "docs.google.test") {
        return json({
          body: {
            content: [
              {
                paragraph: {
                  elements: [{ textRun: { content: "Native document" } }],
                },
              },
            ],
          },
          documentId: "doc-1",
          title: "Document",
        });
      }
      throw new Error(`unexpected Google Workspace request ${url}`);
    },
    tokenProvider: { accessToken: async () => "fixture-bearer-token" },
  });
}

test("Google Workspace CLI exposes native guidance and refuses ambient operational configuration", async () => {
  const output: string[] = [];
  expect(
    await runGoogleWorkspaceCli(["setup", "guidance"], {
      writeLine: (line) => {
        output.push(line);
      },
    }),
  ).toBe(0);
  expect(JSON.parse(output[0] ?? "{}")).toMatchObject({
    security: expect.stringContaining("does not discover"),
  });

  output.length = 0;
  expect(
    await runGoogleWorkspaceCli(["gmail", "labels"], {
      writeLine: (line) => {
        output.push(line);
      },
    }),
  ).toBe(2);
  expect(output[0]).toContain("caller-provided");

  output.length = 0;
  expect(
    await runGoogleWorkspaceCli([], {
      writeLine: (line) => {
        output.push(line);
      },
    }),
  ).toBe(0);
  expect(output).toEqual([GOOGLE_WORKSPACE_CLI_USAGE]);
});

test("Google Workspace CLI maps every documented legacy service action through an explicit native adapter", async () => {
  const output: string[] = [];
  const requests: URL[] = [];
  let adapterPath = "";
  const dependencies: GoogleWorkspaceCliDependencies = {
    loadAdapter: (path) => {
      adapterPath = path;
      return { client: nativeCliClient(requests) };
    },
    writeLine: (line) => {
      output.push(line);
    },
  };
  const adapter = ["--adapter", "./fixture.google-workspace.ts"];
  const commands = [
    ["gmail", "search", "is:unread", "--max", "2"],
    ["gmail", "get", "message-1"],
    [
      "gmail",
      "send",
      "--to",
      "recipient@example.test",
      "--subject",
      "Status",
      "--body",
      "Ready",
      "--cc",
      "copy@example.test",
      "--html",
      "--thread-id",
      "thread-1",
    ],
    [
      "gmail",
      "reply",
      "message-1",
      "--body",
      "Thanks",
      "--from",
      "Agent <agent@example.test>",
    ],
    ["gmail", "labels"],
    [
      "gmail",
      "modify",
      "message-1",
      "--add-labels",
      "STARRED",
      "--remove-labels",
      "UNREAD",
    ],
    [
      "calendar",
      "list",
      "--start",
      "2026-01-01T00:00:00Z",
      "--end",
      "2026-01-02T00:00:00Z",
      "--max",
      "2",
      "--calendar",
      "primary",
    ],
    [
      "calendar",
      "create",
      "--summary",
      "Planning",
      "--start",
      "2026-01-01T10:00:00Z",
      "--end",
      "2026-01-01T11:00:00Z",
      "--location",
      "Office",
      "--description",
      "Review",
      "--attendees",
      "one@example.test,two@example.test",
      "--calendar",
      "primary",
    ],
    ["calendar", "delete", "event-1", "--calendar", "primary"],
    ["drive", "search", "quarterly", "report", "--raw-query", "--max", "2"],
    ["contacts", "list", "--max", "2"],
    ["sheets", "get", "sheet-1", "Sheet1!A1:B1"],
    ["sheets", "update", "sheet-1", "Sheet1!A1:B1", "--values", '[["one",2]]'],
    ["sheets", "append", "sheet-1", "Sheet1!A:C", "--values", '[["two",3]]'],
    ["docs", "get", "doc-1"],
  ];

  for (const command of commands) {
    expect(
      await runGoogleWorkspaceCli([...adapter, ...command], dependencies),
    ).toBe(0);
  }

  expect(adapterPath).toBe("./fixture.google-workspace.ts");
  expect(output).toHaveLength(commands.length);
  expect(JSON.parse(output[0] ?? "[]")).toEqual([]);
  expect(JSON.parse(output.at(-1) ?? "{}")).toMatchObject({
    body: "Native document",
    documentId: "doc-1",
  });
  expect(
    requests.some((url) => url.hostname === "gmail.google.test"),
  ).toBeTrue();
  expect(
    requests.some((url) => url.hostname === "calendar.google.test"),
  ).toBeTrue();
  expect(
    requests.some((url) => url.hostname === "drive.google.test"),
  ).toBeTrue();
  expect(
    requests.some((url) => url.hostname === "people.google.test"),
  ).toBeTrue();
  expect(
    requests.some((url) => url.hostname === "sheets.google.test"),
  ).toBeTrue();
  expect(
    requests.some((url) => url.hostname === "docs.google.test"),
  ).toBeTrue();
});

test("Google Workspace CLI setup does not emit OAuth secrets or create implicit storage", async () => {
  const storage = new MemoryGoogleStorage();
  const output: string[] = [];
  const opened: string[] = [];
  const dependencies: GoogleWorkspaceCliDependencies = {
    browser: {
      open: async (url) => {
        opened.push(url);
      },
    },
    fetchImplementation: async (input) => {
      const url = new URL(String(input));
      if (url.pathname.endsWith("/token")) {
        return json({
          access_token: "fixture-access-token",
          expires_in: 3_600,
          refresh_token: "fixture-refresh-token",
          scope: "scope.one",
        });
      }
      return new Response(null, { status: 200 });
    },
    oauthConfig: {
      authorizationEndpoint: "https://accounts.google.test/authorize",
      clientId: "fixture-client-id",
      clientSecret: "fixture-client-secret",
      redirectUri: "http://127.0.0.1:4567/callback",
      scopes: ["scope.one"],
      tokenEndpoint: "https://accounts.google.test/token",
      revocationEndpoint: "https://accounts.google.test/revoke",
    },
    storage,
    writeLine: (line) => {
      output.push(line);
    },
  };

  expect(
    await runGoogleWorkspaceCli(["setup", "begin", "--open"], dependencies),
  ).toBe(0);
  expect(opened).toHaveLength(1);
  expect(JSON.parse(output[0] ?? "{}")).toEqual({
    authorization_url: opened[0],
  });
  expect(output[0]).not.toContain("codeVerifier");
  expect(output[0]).not.toContain('state"');

  const state = storage.pending?.state;
  expect(state).toBeTruthy();
  expect(
    await runGoogleWorkspaceCli(
      [
        "setup",
        "complete",
        `http://127.0.0.1:4567/callback?code=fixture-code&state=${encodeURIComponent(state ?? "")}`,
      ],
      dependencies,
    ),
  ).toBe(0);
  const completed = output.at(-1) ?? "";
  expect(JSON.parse(completed)).toMatchObject({
    status: { state: "authorized" },
  });
  expect(completed).not.toContain("fixture-access-token");
  expect(completed).not.toContain("fixture-refresh-token");

  expect(await runGoogleWorkspaceCli(["setup", "revoke"], dependencies)).toBe(
    0,
  );
  expect(JSON.parse(output.at(-1) ?? "{}")).toEqual({ revoked: true });
  expect(storage.token).toBeUndefined();
});
