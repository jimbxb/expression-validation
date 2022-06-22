# Expression Validation Solver

A tool to grade written expressions, checking if the required operations are used.

## Example:

```sh
> cat ./example.txt
(a) 'foo', using string concatenation and string slicing 
print('foo'[:] + '')

(b) 1, using len and dictionary lookup and .index
print({3: "ab"}[len("foo")].index("b"))

(c) 2.0, using // and +
1 + 1 / 1

(d) False, using and and 1
1 and False
```

```py
>>> import marker as m
>>> m.check_file("./example.txt", [ \
...     ("foo", {"concat": m.STR_CONCAT, "slice": m.STR_SLICE}), \
...     (1, {"len": m.LEN, "dict lookup": m.DICT_LOOKUP, "index": m.INDEX}), \
...     (2.0, {"//": m.FLOOR_DIV, "+": m.ADD}), \
...     (False, {"and": m.boolop("and"), "1": m.constant(1)}) \
... ])
{0: (1, ['print statement']), 1: (2, []), 2: (1, ["'//' unused"]), 3: (2, [])}
```