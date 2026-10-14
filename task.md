Review and update the automated tests after adding contract C-004 to `data/contracts.json`.

Context:
- The project originally contained three synthetic contracts.
- The current dataset contains four contracts: C-001 through C-004.
- Contract C-004 also involves Harbor Labs.
- Some integration tests still expect only C-002 and C-003 when searching for Harbor Labs.

Tasks:

1. Inspect `data/contracts.json` and the existing tests.
2. Run the full unittest suite and identify all failures.
3. Update outdated test expectations to match the current dataset.
4. Preserve tests that intentionally use mocked data or fixed fixtures.
5. Do not weaken assertions or remove tests merely to make them pass.
6. Do not modify application code unless a genuine implementation bug is discovered; report such bugs separately.
7. Run the full test suite again:
   `uv run --locked python -m unittest discover -s tests -v`
8. Update the testing section of `README.md` if it contains outdated information about failing tests.
9. Do not commit or push changes.

Keep changes minimal and report which tests were updated, why, and the final test results.