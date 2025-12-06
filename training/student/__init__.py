"""
Student model training modules for NeuroPBR.
"""

from .convattn_student import (
    ConvAttnStudentGenerator,
    ConvAttnDistillationLoss,
    create_convattn_student,
)

__all__ = [
    "ConvAttnStudentGenerator",
    "ConvAttnDistillationLoss",
    "create_convattn_student",
]
