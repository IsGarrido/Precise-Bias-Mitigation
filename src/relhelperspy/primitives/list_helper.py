class ListHelper:
    def __init__(self) -> None:
        pass

    def unique(source:list, sort: bool = False):
        items = list(dict.fromkeys(source))

        if sort:
            items.sort()

        return items

    def list_as_file(source:list, sort: bool = True):
        items = ListHelper.unique(source, sort)
        return "\n".join(items)

    def list_as_str_list(source:list):
        return list(map(str, source))

    def apply( items: list, fn ):
        return [fn(item) for item in items]
    
    def filter( items: list, fn ):
        return [item for item in items if fn(item)]

    def sort( items: list, desc = False ):
        items.sort(reverse=desc)
        return items
    
    def to_lower( items: list ):
        return ListHelper.apply(items, str.lower)
    
    def as_lookup(items: 'list[str]'):
        return dict.fromkeys(items, True)

