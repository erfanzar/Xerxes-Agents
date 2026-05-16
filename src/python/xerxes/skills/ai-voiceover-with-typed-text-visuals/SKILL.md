---
name: ai-voiceover-with-typed-text-visuals
description: "AI voiceover with typed text visuals"
version: 0.1.0
tags: [ReadFile, GrepTool, ExecuteShell]
required_tools: [ReadFile, GrepTool, ExecuteShell]
---
# When to use

Apply this skill for tasks similar to: *AI voiceover with typed text visuals*

# Procedure

1. **ReadFile** — file_path=/Users/erfan/.xerxes/skills..., max_chars=3000
2. **GrepTool** — pattern=TTS|ElevenLabs|text.*overla..., path=/Users/erfan/.xerxes/skills..., context=5
3. **GrepTool** — pattern=TTS|text.*overlay|voice|narrat, path=/Users/erfan/.xerxes/skills..., context=20
4. **GrepTool** — pattern=text|quote|lyric|type, path=/Users/erfan/.xerxes/skills..., context=15
5. **ExecuteShell** — command=cat /Users/erfan/.xerxes/sk...
6. **ReadFile** — file_path=/Users/erfan/.xerxes/skills...

# Verification

After running the procedure, the agent should have invoked these tools in order: `ReadFile>GrepTool>GrepTool>GrepTool>ExecuteShell>ReadFile`.
Total successful calls expected: **6**.
Reference final response (truncated): *The **TTS narration** mode is confirmed — text quotes + AI voiceover + typewriter text visuals.  To design this properly, I need to know:  1. **What text/quotes*
