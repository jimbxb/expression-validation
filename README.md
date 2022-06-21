# Expression Validation Solver

A tool to grade written expressions, checking if the required operations are used.

## Example:

```sh
> cat ./example.txt
(a) 'foo', using string concatenation and string slicing 
'foo'[:] + ''

(b) 1, using len and dictionary lookup 
len({'foo': "a"}["foo"])

(c) 2.0, using // and +
1 + 1 / 1

(d) False, using and and 1
1 and False
```

```
>>> marker.check_file("./example.txt", [("foo", {"concat": marker.STR_CONCAT, "slice": marker.STR_SLICE}), \
...     (1, {"len": marker.LEN, "dict lookup": marker.DICT_LOOKUP}), \
...     (2.0, {"//": marker.INT_DIV, "+": marker.ADD}), \
...     (False, {"and": marker.boolop("and"), "1": marker.constant(1)})])
{0: (2, []), 1: (2, []), 2: (1, ["'//' unused"]), 3: (2, [])}
```