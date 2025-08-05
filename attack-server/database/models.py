from sqlmodel import SQLModel, Field

class Job(SQLModel, table=True):
    id: int = Field(default=None,primary_key=True, index=True)
    progress : float
    last_attack_performed : str | None
    is_over : bool = Field(default=False)
    dataset : str
    model : str