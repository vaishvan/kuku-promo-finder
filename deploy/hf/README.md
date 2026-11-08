---
title: Promo Finder
emoji: 🎬
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
short_description: Finds and ranks promo clips in microdrama screenplays
---

# Promo Finder

Upload a microdrama screenplay zip (or click **Try the sample show**). The tool scores every 20–60 second cut,
sends a shortlist to an AI judge for hook, cliffhanger and standalone clarity, and returns ranked promos with
timecodes, a posting plan, and a learning loop that retrains from your Accept/Reject choices.

Source and write-up: https://github.com/vaishvan/kuku-promo-finder

Uploads are processed on this server and deleted after 6 hours; shortlisted clips are scored by NVIDIA's AI API.
