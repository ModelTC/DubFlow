# S3FD detector from TalkNet-ASD

Only `model/faceDetector/` is vendored from
[TalkNet-ASD](https://github.com/TaoRuijie/TalkNet-ASD), revision
`6d6821479af485e251c4991487e40573b42181b4`.

The detector receives an explicit checkpoint path instead of downloading weights
at import time. The network, NMS and forward calculation are unchanged; see
`../patches/talknet-external-weights.patch`.

TalkNet audio/visual encoders, attention, training loss and training dependencies
are not used. Active-speaker detection is provided separately by UniTalk-LoCoNet.

Retain [LICENSE.md](LICENSE.md) and upstream attribution when redistributing.
