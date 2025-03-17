from pydantic import BaseModel, ValidationError

class User(BaseModel):
    id: int
    name: str
    age: int = 0  # 默认值

try:
    user = User(id="invalid", name="小明")  # id 非整数会报错
except ValidationError as e:
    print(e)  # 输出：1 validation error for User → id → 值不是有效整数 [1](@ref)[6](@ref)
