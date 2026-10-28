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

- **Standard:** approximately 265 MB once: Silero VAD v5 (fp32, 2.2 MB), Moonshine Streaming
  small q8 (166 MB), Kokoro-82M q8, and the English `af_heart` voice. On 13 real-speech clips in
  Chrome/WASM, Moonshine Streaming small made 5.5% word errors versus 6.8% for Moonshine base.
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
Before a call the phone checks which model files are already in its browser caches. If all of them
are there, the call view shows **Starting voice** while it loads them into memory, with no
download. Otherwise it shows **Downloading voice (X of Y MB)**, where Y counts only the files that
are still missing. On the first call the app asks the browser to keep this storage
(`navigator.storage.persist()`). That makes eviction less likely, but the browser may still
refuse.
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

If on-phone WASM recognition fails (including a bare numeric ONNX error), that utterance is
**dropped**, never replayed or sent as a partial message. The call briefly shows “Didn't catch
that — restoring speech…” and **replaces the whole speech worker**, reloading both Moonshine
and Kokoro from the browser’s model cache. Recreating a pipeline in the same worker is not
sufficient: an ONNX failure can poison the runtime. VAD and the microphone stay open; input
received during the short reload is discarded, and “Didn't catch that — say it again” appears
when listening resumes. A loading/progress state is shown only if recovery takes more than
750 ms. Speech output uses the freshly reloaded Kokoro, not the failed runtime.

The third recovery request within a rolling 60 seconds ends the call with a clear suggestion
to choose **Light**. A reload that fails, changes tier, or takes more than 30 seconds also ends
the call with a Light-mode error. End/cancellation kills an in-progress replacement worker;
late worker results cannot submit a transcript or restart a call after hang-up.

Code blocks, URLs and Markdown markup are omitted from speech. Long replies are shortened to
4,000 characters, spoken in bounded sentences as audio becomes ready; the complete response
stays on screen. Interrupting cancels local playback and pending synthesis, **not** the task or
supervisor run. Permission approvals remain human-only and are never answered by call mode.

## Inspect what speech recognition actually heard

If recognition is inaccurate, open the voice **settings gear** and enable **Keep last utterance
for debugging**. It is **off by default**; only that preference is remembered in localStorage.
After the next utterance, the settings panel shows the raw transcript, exact duration/sample
count, unnormalised peak and RMS levels (1 = full scale), and VAD start/end offsets. Offsets
are measured from the beginning of the captured clip: start includes the retained pre-roll;
end is exclusive and excludes the trailing silence used to finish the turn. The audio itself
still contains both pre-roll and trailing silence, exactly as passed to transcription.

**Play** uses Web Audio to play those original 16 kHz Float32 samples without normalising or
re-encoding them. It interrupts reply speech and temporarily mutes the live microphone so the
clip cannot trigger another voice turn; stopping/finishing playback restores an unmuted mic if
it was unmuted before. Use headphones to inspect quiet speech, clipping, noise or cut-off words.
**Save WAV** creates a 16-bit, mono, 16 kHz PCM WAV locally. On phones supporting file sharing it
opens the system share sheet so you choose the destination; otherwise it downloads
`xerxes-utterance.wav` and revokes the temporary download URL afterwards. Cancelling the share
sheet does not start a download. WAV conversion quantises/clips to 16-bit PCM; Play and the
reported levels use the untouched Float32 model input.

Only the **last utterance**, at most **30 seconds** (480,000 Float32 samples, about 1.9 MB), is
retained in memory. A new utterance replaces it; **End**, a call error, turning the toggle off,
or reloading/leaving the app clears it. Turning the toggle off during transcription cannot
restore that clip when recognition finishes. Audio is never put in localStorage, browser
caches, the chat thread, logs or an automatic upload. Only you can explicitly export it with
Save WAV; recognised text still follows the normal chat behaviour described above.

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
