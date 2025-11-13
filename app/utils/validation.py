from datetime import date
from uuid import UUID, uuid4
from enum import Enum
from pydantic import BaseModel
from pandantic import Pandantic
from typing import Optional


# Membuat class fraud untuk validasi pake pydantics
class Fraud(BaseModel):
    amount: float
    location: str
    is_fraud: Optional[int] = None
