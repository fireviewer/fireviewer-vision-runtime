"""Part.2 interchangeable managed and baseline VisionProvider implementations."""

from fireviewer_vision_runtime.mvp.vision.bedrock_nova import (
    BedrockImage,
    BedrockImageLoader,
    BedrockNovaVisionConfig,
    BedrockNovaVisionProvider,
    BedrockRuntimeClient,
)
from fireviewer_vision_runtime.mvp.vision.event_vision import (
    EventVisionConfig,
    EventVisionRun,
    EventVisionRunner,
    VisionArtifact,
    vision_result_reference,
)
from fireviewer_vision_runtime.mvp.vision.grounding_dino import (
    GroundingDinoConfig,
    GroundingDinoVisionProvider,
    VisionImageLoader,
)
from fireviewer_vision_runtime.mvp.vision.local_grounding_dino_bundle import (
    LocalGroundingDinoBundleManifest,
    LocalGroundingDinoFile,
    LocalGroundingDinoModelLoader,
    inspect_local_grounding_dino_bundle,
)
from fireviewer_vision_runtime.mvp.vision.video_keyframes import (
    DecodedVideoFrame,
    DecodedVideoSample,
    OpenCvVideoFrameDecoder,
    VideoFrameDecoder,
    VideoKeyframeArtifact,
    VideoKeyframeConfig,
    VideoKeyframeExtractor,
    VideoKeyframeRun,
)
from fireviewer_vision_runtime.mvp.vision.yolo import (
    HuggingFaceYoloModelLoader,
    YoloCpuConfig,
    YoloCpuVisionProvider,
)

__all__ = [
    "BedrockImage",
    "BedrockImageLoader",
    "BedrockNovaVisionConfig",
    "BedrockNovaVisionProvider",
    "BedrockRuntimeClient",
    "DecodedVideoFrame",
    "DecodedVideoSample",
    "EventVisionConfig",
    "EventVisionRun",
    "EventVisionRunner",
    "GroundingDinoConfig",
    "GroundingDinoVisionProvider",
    "HuggingFaceYoloModelLoader",
    "LocalGroundingDinoBundleManifest",
    "LocalGroundingDinoFile",
    "LocalGroundingDinoModelLoader",
    "OpenCvVideoFrameDecoder",
    "VideoFrameDecoder",
    "VideoKeyframeArtifact",
    "VideoKeyframeConfig",
    "VideoKeyframeExtractor",
    "VideoKeyframeRun",
    "VisionArtifact",
    "VisionImageLoader",
    "YoloCpuConfig",
    "YoloCpuVisionProvider",
    "inspect_local_grounding_dino_bundle",
    "vision_result_reference",
]
