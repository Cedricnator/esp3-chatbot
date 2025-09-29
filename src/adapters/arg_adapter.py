from abc import ABC, abstractmethod
from argparse import Namespace

class ArgAdapter(ABC):
  def __init__(self) -> None:
    pass

  @abstractmethod
  def parse(self) -> Namespace:
    pass