#!/usr/bin/env bun
import { render } from "ink";
import { App } from "./App.js";
import { parseArgs } from "./utils/args.js";

function main() {
  // eslint-disable-next-line no-console
  console.error("[xerxes-build] 2026-04-21T14:20Z");
  const args = parseArgs(process.argv.slice(2));
  if (!args) process.exit(2);

  // Subcommands (wakeup / send) are deferred — MVP focuses on interactive mode.
  if (args.command === "wakeup" || args.command === "send") {
    console.error(`'${args.command}' subcommand not yet implemented in the TS CLI.`);
    process.exit(2);
  }

  if (args.print) {
    console.error("Non-interactive (-p / --print) mode not yet implemented.");
    process.exit(2);
  }

  const { unmount, waitUntilExit } = render(<App args={args} />, {
    exitOnCtrlC: false,
  });

  let sigintHandled = false;
  process.on("SIGINT", () => {
    if (sigintHandled) return;
    sigintHandled = true;
    setTimeout(() => {
      unmount();
      process.exit(0);
    }, 100);
  });

  waitUntilExit().then(() => {
    if (sigintHandled) return;
    sigintHandled = true;
    process.exit(0);
  });
}

main();
