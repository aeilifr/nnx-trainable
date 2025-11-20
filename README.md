# NNX Trainable
A parent class for NNX models that are meant to be trained, evaluated, and checkpointed.

```python
class Trainable(ABC, nnx.Module):

  def save(self, **kwargs):...

  @classmethod
  def load(cls, dir: str, **kwargs):...

  def log(self, metrics: Dict, **kwargs):...

  @abstractmethod
  def loss(self, batch: Dict, **kwargs):...

  @abstractmethod
  def evaluate(self, batch: Dict, **kwargs):...

  def fit(self, **kwargs):...
```
