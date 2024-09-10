from typing import Generic, TypeVar, Optional

T = TypeVar('T')
E = TypeVar('E')

class RelResult(Generic[T, E]):
    def __init__(self, success_result: Optional[T] = None, error_result: Optional[E] = None) -> None:
        if success_result is not None and error_result is not None:
            raise ValueError("RelResult cannot have both a success and an error result.")
        self.success_result: Optional[T] = success_result
        self.error_result: Optional[E] = error_result

    def ok(self) -> bool:
        return self.success_result is not None

    def error(self) -> bool:
        return self.error_result is not None

    def get_success(self) -> Optional[T]:
        return self.success_result

    def has_error(self) -> bool:
        return self.error_result is not None
    
    def get_error(self) -> Optional[E]:
        return self.error_result

    @staticmethod
    def success(success_result: T) -> 'RelResult[T, None]':
        return RelResult(success_result=success_result)

    @staticmethod
    def error(error_result: E) -> 'RelResult[None, E]':
        print(f"Error: {error_result}")
        return RelResult(error_result=error_result)

    def raise_if_error(self):
        if self.error_result is not None:
            raise Exception(self.error_result)

    def __repr__(self) -> str:
        if self.ok():
            return f"RelResult(success: {self.success_result})"
        else:
            return f"RelResult(error: {self.error_result})"
