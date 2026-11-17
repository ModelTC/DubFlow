# Inference snapshot

Upstream: https://github.com/SJTUwxz/LoCoNet_ASD at
`68d90c83fde956d60245d0715bbd18e9e5fd3bae`.
The ordinary UniTalk-LoCoNet checkpoint is published by
https://github.com/plnguyen2908/UniTalk-ASD-code (not the TalkNCE variant).

- Kept encoder, attention, convolution, visual frontend, VGGish network and
  log-mel feature implementation. Omitted training and data loaders.
- Changed imports to package-relative paths to coexist with S3FD.
- Instantiate `vggish._vgg()` directly. The complete ASD checkpoint contains
  all encoder parameters; strict loading replaces pretrained initialization.
- Removed unused VGGish download/postprocessing wrappers. Network operations
  and checkpoint tensor names remain unchanged.

No repository-level LICENSE was present in this upstream snapshot. Per-file
copyright/license notices are preserved. This is not a claim that the entire
LoCoNet model is Apache-2.0; obtain applicable rights before redistribution.
