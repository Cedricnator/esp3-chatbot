import pandas as pd
import time
from utils.calculator import Calculator

class Checkpoint:  
  def __init__(self, last_checkpoint_time: float, name: str) -> None:
    self._time = self.makeTime(last_checkpoint_time) 
    self._name = name
  
  def makeTime(self, last_checkpoint_time: float):
    return round(time.time() - last_checkpoint_time, 2)  

  def get_name(self) -> str:
    return self._name
  
  def get_time(self) -> float:
    return self._time

class CheckpointerRegister:
  def __init__(self) -> None:
    self.checkpoints = []
    self.starting_time = time.time()
    self.checkpoint_path = "data/checkpoints.csv" 
    self.last_checkpoint_time = self.starting_time
    self.totalCost = 0.0
    try:
      self.checkpoint_df = pd.read_csv(self.checkpoint_path)
    except (FileNotFoundError, pd.errors.EmptyDataError):      
      self.checkpoint_df = pd.DataFrame()

  def setCheckpoint(self, name: str) -> None:
    self.checkpoints.append(Checkpoint(self.last_checkpoint_time, name))
    self.last_checkpoint_time = time.time()
  
  def save(self) -> None:
    # Build a single-row mapping: { checkpoint_name: checkpoint_time, ... }
    row = {str(c.get_name()): c.get_time() for c in self.checkpoints}
    if not row:
      return
    row["total_time"] = round(time.time() - self.starting_time, 2)
    row["total_cost (US)"] = self.totalCost
    new_row_df = pd.DataFrame([row])
    # Concatenate, allowing new columns to be added if necessary
    self.checkpoint_df = pd.concat([self.checkpoint_df, new_row_df], ignore_index=True, sort=False)
    self.checkpoint_df.to_csv(self.checkpoint_path, index=False)
    # Reset in-memory checkpoints after persisting
    self.checkpoints = []

  def addCost(self, raw: str) -> None:
    self.totalCost = Calculator.calculate_total_tost(raw)


