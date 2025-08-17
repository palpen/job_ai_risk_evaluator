# Python Application - Claude Instructions

## Code Style Guidelines
- Follow PEP 8 style guide for Python code
- Use descriptive variable and function names (snake_case)
- Keep functions small and focused on a single responsibility
- Add docstrings to all functions and classes
- Use type hints for function parameters and return values
- Prefer explicit imports over wildcard imports

## Project Structure
- Keep modules organized and logically grouped
- Separate concerns: models, views, utilities, tests
- Use meaningful file and directory names
- Include proper `__init__.py` files for packages

## Testing Guidelines
- Write unit tests for all functions and classes
- Use pytest as the testing framework
- Aim for high test coverage
- Test edge cases and error conditions
- Keep tests simple and readable

## Common Commands
- `python -m pytest`: Run all tests
- `python -m pytest tests/test_specific.py`: Run specific test file
- `python -m black .`: Format code with Black
- `python -m flake8`: Check code style
- `python -m mypy .`: Type checking

## Dependencies
- Use requirements.txt or pyproject.toml for dependency management
- Pin dependency versions for reproducibility
- Separate dev dependencies from production dependencies

## Error Handling
- Use specific exception types rather than generic Exception
- Handle errors gracefully with appropriate error messages
- Log errors appropriately for debugging

## Documentation
- Keep README.md updated with setup and usage instructions
- Document complex logic with inline comments
- Use clear commit messages following conventional commits format