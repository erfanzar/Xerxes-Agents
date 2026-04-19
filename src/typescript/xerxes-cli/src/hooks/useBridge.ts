import { useEffect, useRef } from "react";
import type { Bridge } from "../bridge/spawn.js";
import { spawnBridge } from "../bridge/spawn.js";
import type { Event, Request } from "../bridge/types.js";

interface UseBridgeOpts {
  python: string;
  projectDir: string;
  onEvent: (event: Event) => void;
}

export function useBridge(opts: UseBridgeOpts): {
  send: (req: Request) => void;
} {
  const ref = useRef<Bridge | null>(null);
  const onEventRef = useRef(opts.onEvent);
  onEventRef.current = opts.onEvent;

  useEffect(() => {
    const bridge = spawnBridge({
      python: opts.python,
      projectDir: opts.projectDir,
    });
    ref.current = bridge;
    const unsub = bridge.on((e) => onEventRef.current(e));
    return () => {
      unsub();
      bridge.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return {
    send: (req) => ref.current?.send(req),
  };
}
