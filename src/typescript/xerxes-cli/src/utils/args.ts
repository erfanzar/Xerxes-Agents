export interface CliArgs {
  model: string;
  baseUrl?: string;
  apiKey?: string;
  python: string;
  projectDir: string;
  permissionMode: "auto" | "accept-all" | "manual";
  yolo: boolean;
  print?: string;
  command?: "wakeup" | "send";
  commandArgs: string[];
}

const HELP = `Xerxes — agent CLI

Usage: xerxes [OPTIONS] [COMMAND] [ARGS...]

Commands:
  wakeup              Start the background daemon
  send PROMPT         Send a task to the running daemon

Options:
  -m, --model MODEL           Model name (e.g. gpt-4o, claude-sonnet-4-6)
      --base-url URL          API base URL
      --api-key KEY           API key for the provider
      --python CMD            Python executable (default: python)
      --project-dir PATH      Project directory (default: cwd)
      --permission-mode MODE  auto | accept-all | manual (default: auto)
      --yolo                  Alias for --permission-mode accept-all
  -p, --print PROMPT          Non-interactive mode: print and exit
  -h, --help                  Show this help
  -v, --version               Show version
`;

export function parseArgs(argv: string[]): CliArgs | null {
  const out: CliArgs = {
    model: "",
    python: "python",
    projectDir: process.cwd(),
    permissionMode: "auto",
    yolo: false,
    commandArgs: [],
  };

  const need = (i: number, name: string): string => {
    const v = argv[i + 1];
    if (v === undefined) {
      console.error(`Missing value for ${name}`);
      process.exit(2);
    }
    return v;
  };

  let i = 0;
  while (i < argv.length) {
    const a = argv[i];
    if (a === "-h" || a === "--help") {
      console.log(HELP);
      process.exit(0);
    } else if (a === "-v" || a === "--version") {
      console.log("xerxes 0.2.0");
      process.exit(0);
    } else if (a === "-m" || a === "--model") {
      out.model = need(i, "--model");
      i += 2;
    } else if (a === "--base-url") {
      out.baseUrl = need(i, "--base-url");
      i += 2;
    } else if (a === "--api-key") {
      out.apiKey = need(i, "--api-key");
      i += 2;
    } else if (a === "--python") {
      out.python = need(i, "--python");
      i += 2;
    } else if (a === "--project-dir") {
      out.projectDir = need(i, "--project-dir");
      i += 2;
    } else if (a === "--permission-mode") {
      const v = need(i, "--permission-mode");
      if (v !== "auto" && v !== "accept-all" && v !== "manual") {
        console.error(`Invalid --permission-mode: ${v}`);
        process.exit(2);
      }
      out.permissionMode = v;
      i += 2;
    } else if (a === "--yolo") {
      out.yolo = true;
      out.permissionMode = "accept-all";
      i += 1;
    } else if (a === "-p" || a === "--print") {
      out.print = need(i, "--print");
      i += 2;
    } else if (a === "wakeup" || a === "send") {
      out.command = a;
      out.commandArgs = argv.slice(i + 1);
      break;
    } else if (a !== undefined && a.startsWith("-")) {
      console.error(`Unknown option: ${a}`);
      process.exit(2);
    } else {
      // Positional — treat as prompt for non-interactive mode if -p wasn't given.
      if (!out.print && a !== undefined) out.print = a;
      i += 1;
    }
  }

  return out;
}
