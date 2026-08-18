"""Chatterbox S3Gen one-step flow-matching: teacher (10-step CFM) + distilled student."""

from dsflow.chatterbox.model import OneStepS3Gen, load_teacher

__all__ = ["OneStepS3Gen", "load_teacher"]
