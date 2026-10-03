// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * POSIX shell that installs Bun into `~/.bun/bin` on a remote host without
 * Bun's own installer, which requires curl and bash. It needs only one of
 * curl or wget and one of unzip or bsdtar, chooses the official build for
 * the host (OS, CPU, musl, AVX2 baseline, Rosetta), and checks the download
 * against Bun's published SHA-256 sums whenever the host can compute one.
 *
 * Detection lives in small functions so a host quirk is a one-line change
 * and tests can override any of them after sourcing the script.
 * `xerxes_install_bun` returns 0 on success; on failure it prints one
 * actionable line on stdout and returns non-zero.
 */
export const BUN_RELEASES = 'https://github.com/oven-sh/bun/releases/latest/download'

export function remoteBunInstallScript(releases: string = BUN_RELEASES): string {
  const quoted = `'${releases.replaceAll("'", "'\\''")}'`
  return `xerxes_bun_releases=${quoted}
xerxes_fetch() {
  if command -v curl >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 --connect-timeout 15 --max-time 300 -fsSL "$1" -o "$2"
  elif command -v wget >/dev/null 2>&1; then
    # Only options GNU and BusyBox wget share; the URL itself is https.
    wget -q -T 30 -O "$2" "$1"
  else
    return 127
  fi
}
xerxes_has_avx2() {
  case "$(uname -s)" in
    Linux) grep -qw avx2 /proc/cpuinfo 2>/dev/null ;;
    Darwin) sysctl -n machdep.cpu.leaf7_features 2>/dev/null | grep -qi avx2 ;;
    *) return 1 ;;
  esac
}
xerxes_is_musl() { [ -f /etc/alpine-release ] || ldd --version 2>&1 | grep -qi musl; }
xerxes_is_rosetta() { [ "$(sysctl -n sysctl.proc_translated 2>/dev/null)" = 1 ]; }
xerxes_bun_asset() {
  case "$(uname -s)" in Linux) os=linux ;; Darwin) os=darwin ;; *) return 1 ;; esac
  case "$(uname -m)" in x86_64|amd64) arch=x64 ;; aarch64|arm64) arch=aarch64 ;; *) return 1 ;; esac
  # An x64 shell under Rosetta still runs the faster native arm64 build.
  if [ "$os" = darwin ] && [ "$arch" = x64 ] && xerxes_is_rosetta; then arch=aarch64; fi
  libc=''
  if [ "$os" = linux ] && xerxes_is_musl; then libc=-musl; fi
  baseline=''
  if [ "$arch" = x64 ] && ! xerxes_has_avx2; then baseline=-baseline; fi
  printf 'bun-%s-%s%s%s' "$os" "$arch" "$libc" "$baseline"
}
xerxes_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d ' ' -f 1
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d ' ' -f 1
  elif command -v openssl >/dev/null 2>&1; then openssl dgst -sha256 "$1" | sed 's/.*= *//'
  else return 127
  fi
}
xerxes_unzip() {
  if command -v unzip >/dev/null 2>&1; then unzip -oq "$1" -d "$2"
  elif command -v bsdtar >/dev/null 2>&1; then bsdtar -xf "$1" -C "$2"
  else return 127
  fi
}
xerxes_install_bun() {
  if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
    echo 'Install curl or wget on this host, then reconnect.'; return 1
  fi
  if ! command -v unzip >/dev/null 2>&1 && ! command -v bsdtar >/dev/null 2>&1; then
    echo 'Install unzip on this host, then reconnect.'; return 1
  fi
  asset=$(xerxes_bun_asset) || { echo "Bun has no build for this host ($(uname -s) $(uname -m))."; return 1; }
  work=$(mktemp -d "\${TMPDIR:-/tmp}/xerxes-bun.XXXXXX") || { echo 'Could not create a temporary folder for Bun.'; return 1; }
  xerxes_fetch "$xerxes_bun_releases/$asset.zip" "$work/$asset.zip" || { rm -rf "$work"; echo 'Could not download Bun. Check that this host can reach github.com.'; return 1; }
  if digest=$(xerxes_sha256 "$work/$asset.zip"); then
    xerxes_fetch "$xerxes_bun_releases/SHASUMS256.txt" "$work/SHASUMS256.txt" || { rm -rf "$work"; echo 'Could not download the Bun checksums. Check that this host can reach github.com.'; return 1; }
    expected=$(grep " $asset.zip\$" "$work/SHASUMS256.txt" | cut -d ' ' -f 1)
    if [ -z "$expected" ] || [ "$expected" != "$digest" ]; then rm -rf "$work"; echo 'The Bun download did not match its published checksum. Reconnect to try again.'; return 1; fi
  fi
  xerxes_unzip "$work/$asset.zip" "$work" && [ -f "$work/$asset/bun" ] || { rm -rf "$work"; echo 'Could not unpack the Bun download.'; return 1; }
  mkdir -p "$HOME/.bun/bin" &&
    mv -f "$work/$asset/bun" "$HOME/.bun/bin/bun.partial" &&
    chmod 755 "$HOME/.bun/bin/bun.partial" &&
    mv -f "$HOME/.bun/bin/bun.partial" "$HOME/.bun/bin/bun" || { rm -rf "$work"; echo 'Could not install Bun into ~/.bun/bin. Check permissions and free space.'; return 1; }
  rm -rf "$work"
}
`
}
