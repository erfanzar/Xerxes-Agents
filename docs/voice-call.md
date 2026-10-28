# Call XERXES from your phone

Open the phone app’s **Xerxes** tab and tap the round **waveform voice button next to Send**.
The full-screen voice scene shows a glowing orb, a state line and short captions. Tap **Start
voice** and allow microphone access. The small **settings gear** holds the Standard/Light picker
and download sizes; choose a tier before starting. The first-call download notice stays visible
until you start a call.

Speak naturally; after a short pause XERXES answers aloud. The orb reacts to microphone audio
and, in Standard mode, to actual playback levels. Light uses the phone’s built-in voice, whose
PCM levels browsers do not expose, so it uses a speaking-state glow instead. Reduced-motion
preferences disable the rotating/pulsing motion. Captions show what was just heard and the
current spoken sentence. Speak for about a third of a second, or **tap the orb**, to interrupt.

Bottom controls are **Mute**, **Chat**, and the red **End ×** button. **Chat** (or Escape) minimizes
voice to a slim pill at the top of the Xerxes tab **without hanging up**. Voice keeps listening
and speaking; you can read the normal thread and send typed messages. Tap the pill or composer
voice button to return to the orb. Moving between app tabs also keeps the same call alive.
Only **End**, a call error, leaving the app entirely, or hiding the browser tab for 60 seconds
releases the microphone and models. The minimized pill has its own End button.

The supervisor’s ordinary model must already be configured and the phone must be connected
to the Hub over HTTPS. **No Hub speech-provider key or voice-provider configuration is needed.**
The browser needs AudioWorklet, Web Workers and WebAssembly. Browser/device support and
real-time performance vary; Light is recommended for low-memory phones.

## Downloads and memory

- **Standard:** approximately 162 MB once: Silero VAD v5 (fp32, 2.2 MB), Moonshine base q8,
  Kokoro-82M q8, and the English `af_heart` voice.
- **Light:** approximately 34 MB once: Silero VAD v5 and Moonshine tiny q8. Replies use an
  **installed local English phone voice**. If there is no local English voice, install one in
  the phone’s speech settings or use Standard; network speech-synthesis voices are refused.

The picker shows exact model-download totals reported by the Hub. These are download sizes,
not peak RAM: inference needs additional working memory. Devices reporting less than 4 GB of
RAM default to Light. You can choose either tier. If Kokoro cannot initialize, the call switches
to Light (including Moonshine tiny) and displays one notice. All three models run on
single-threaded WASM. Quantized models are not used on WebGPU: in Chrome, q8 Moonshine on WebGPU
decoded garbage, while on WASM it transcribed correctly in about 0.3 s. int8 files are not used
at all, because ONNX Runtime Web has no `ConvInteger` kernel and cannot load them.

Models load only when you start a call. The Hub downloads and verifies pinned files once;
the phone then downloads them from that same Hub. Transformers.js caches model files in the
browser after the first call. Clearing site data, browser storage eviction or a new model
revision requires another download. The app shell service worker does **not** precache models.
The ONNX runtime and phonemizer are also served/bundled locally; no third-party origin is
contacted by the phone. A missing model route or disabled/unavailable Hub model store is
reported in the call view rather than silently contacting another server.

## What is sent and spoken

Microphone audio, turn detection, speech recognition and Standard speech generation all run
on the phone. Silero VAD has its own small-model worker so blocking STT/TTS inference cannot
stall turn detection or barge-in; STT and TTS share a separate speech worker. Both use the same
pinned transformers.js / ONNX assets with independent single-threaded sessions. Only recognized
text is submitted as a normal voice-source message to the XERXES supervisor. Transcripts and full
replies remain in the normal chat thread. This is hands-free: recognized text is sent without
an additional Send tap. Existing push-to-talk still fills an editable draft and works separately.

VAD uses 16 kHz audio, waits at least 250 ms to recognize speech, retains 200 ms of pre-roll,
ends after 700 ms of silence and caps utterances at 30 seconds. After 120 seconds without a
reply, the call says “XERXES is still working; I'll read the reply when it arrives” and returns
to listening without unloading models or replaying the message. A matching late reply is still
spoken unless you have started speaking again. Replies are matched by causal message ID;
older Hubs use the next new non-wake supervisor reply. Autopilot/Watch wake messages are ignored.
The microphone queue drops old frames rather than ending a call when busy. A VAD worker that
fails to answer for 10 seconds ends the call with a specific error.

Code blocks, URLs and Markdown markup are omitted from speech. Long replies are shortened to
4,000 characters, spoken in bounded sentences as audio becomes ready; the complete response
stays on screen. Interrupting cancels local playback and pending synthesis, **not** the task or
supervisor run. Permission approvals remain human-only and are never answered by call mode.

## Troubleshooting

- **Hub does not support call models:** update the Hub and phone bundle together. No runtime
  or deployment is restarted automatically.
- **Call models unavailable/disabled:** ask the Hub owner to check its voice-model setting,
  storage and upstream download access. Retry the call after that is resolved.
- **Microphone/audio blocked:** use HTTPS, allow microphone access, and tap Start again.
  Keep the phone app visible. Headphones reduce speaker echo and accidental interruptions.
- **Too slow or out of memory:** choose Light, close other tabs, or use ordinary push-to-talk.
- **Light cannot speak:** install a local English voice. Cloud/network voices are not used.

Real-phone checks still needed for any new browser/device: microphone permission and autoplay,
echo cancellation, VAD sensitivity in noise, WebGPU/WASM compatibility, warm-cache behavior,
peak memory, sustained inference speed, barge-in latency and background-tab suspension.
