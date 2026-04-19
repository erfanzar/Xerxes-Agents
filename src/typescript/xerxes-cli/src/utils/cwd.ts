import os from "node:os";

export function tildify(path: string): string {
  const home = os.homedir();
  if (home && path.startsWith(home)) {
    return "~" + path.slice(home.length);
  }
  return path;
}
