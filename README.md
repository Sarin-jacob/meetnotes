# MeetNotes

An offline meeting-minutes app for Android phones. It records an in-person meeting, then on the phone itself:

- transcribes it (Hindi/English → English)
- works out who spoke when
- writes a summary and minutes

and exports a Markdown zip. Nothing leaves the device.

## Current stage: device benchmark

Before building the full app, the benchmark page checks whether the team's phones (6–8 GB RAM) can run each step fast enough:

1. **This phone**: WebGPU, RAM, CPU, storage and wake-lock support.
2. **Test audio**: record a 3–5 minute Hindi/English conversation, pick a file, or load a sample. It also checks whether audio is lost when the screen goes off.
3. **Speech → English**: Whisper (base or small) via transformers.js, on GPU or CPU.
4. **Who spoke when**: pyannote segmentation plus WeSpeaker voiceprints, clustered on the device.
5. **Summary & minutes**: a small LLM via WebLLM (Qwen3.5 2B, Gemma, Llama…).
6. **Screen & alerts**: the dark "recording" screen with wake lock, and the handover alarm with vibration.

Each step shows an estimate for a 1-hour meeting. **Copy / Share / Download JSON** gives you the results to send back.

### Testing on a phone

Open the GitHub Pages URL in Chrome on Android. The first run downloads the models (about 250 MB for Whisper small and 1–1.5 GB for the LLM), so use Wi-Fi.

## Development

```bash
npm install
npm run dev        # http://localhost:5173
npm test           # unit tests
npm run typecheck
npm run deploy     # build and publish to GitHub Pages (gh-pages branch)
```
