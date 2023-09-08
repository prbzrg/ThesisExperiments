"""
    dummy_project_function(x, y) → z

Dummy function for illustration purposes.
Performs operation:

```math
z = x + y
```
"""
@inline @fastmath function dummy_project_function(x, y)
    return x + y
end
