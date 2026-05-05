# Docstring Addition Prompt

You are adding comprehensive Google-style docstrings to Python files in the Xerxes project.

## Rules

1. **Add docstrings to EVERY:**
   - Module (top of file, after imports if none exists)
   - Class (after class declaration)
   - Function / Method / Async Function / Staticmethod / Classmethod / Property
   - Abstract method (can be brief)

2. **Do NOT add docstrings for variable assignments like:**
   ```python
   X = ...
   """this is a docstring"""
   ```
   If you see standalone string literals after assignments, leave them as-is (or remove if they exist as old attribute docstrings).

3. **Google-Style Format:**
   ```python
   """One-line summary.

   Longer description if the function/class is complex.

   Args:
       param_name (type): Description. IN: what the argument represents.
           OUT: how it's used / what happens to it.
       *args: Description. IN: ... OUT: ...
       **kwargs: Description. IN: ... OUT: ...
           Expected keys when TypedDict/unpacked type is used:
           - key_name: description
           - key_name2: description

   Returns:
       type: Description. OUT: what the caller receives.

   Raises:
       ExceptionName: When and why it's raised.

   Example:
       >>> result = func(arg1, arg2)
       >>> print(result)
       'expected output'
   """
   ```

4. **IN / OUT for every arg/kwarg:**
   - IN: What value/type the caller should pass.
   - OUT: How the function uses it or what it returns.
   - For `**kwargs`, if the type annotation is a TypedDict or similar, enumerate ALL fields inside the docstring under the kwargs entry.

5. **Update existing docstrings** if they are outdated, incomplete, or missing IN/OUT info.

6. **Do NOT change any code logic.** Only add or update docstrings.

7. **Keep docstrings truthful** — derive descriptions from the actual code, not hallucinations.

8. **For abstract methods** in ABCs, a brief docstring is fine.

9. **For `__init__` methods**, document the parameters that configure the instance.

10. **For properties**, document what the getter returns.

11. **Module-level docstrings** should explain the module's purpose and main exports.

## Style Notes
- Use triple double-quotes `"""`.
- One-line docstrings are OK for trivial functions, but prefer multi-line with Args/Returns for anything non-trivial.
- Match the existing indentation.
