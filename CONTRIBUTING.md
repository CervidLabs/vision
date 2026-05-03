# Contributing to Cervid 🦌

Thank you for your interest in improving the Cervid engine! To maintain our "Military-Grade" infrastructure and code integrity, all contributions must adhere to these strict guidelines.

## 🛡️ Quality Standards (Zero Tolerance)

This project utilizes a CI (Continuous Integration) pipeline that automatically blocks any code that does not meet the following criteria:

1.  **Strict TypeScript:** The use of `any` is strictly prohibited. Everything must be properly typed.
2.  **Linting & Formatting:** Code must pass `eslint` and `prettier` checks. This is enforced locally via Husky before every commit.
3.  **Zero Warnings:** We do not accept Pull Requests that generate warnings in the console. Warnings are treated as errors.

## 📝 Workflow & Discipline

1. **Fork & Branch:** Create a descriptive branch (e.g., `feat/new-evaluator` or `fix/memory-leak`).
2. **Development:** Write clean, documented code following the existing architectural patterns.
3. **Local Validation:** Ensure that `npm run lint` returns no errors before attempting to commit.
4. **Semantic Commits:** We follow **Conventional Commits**. Your commit message must follow this format:
   - `feat: ...` for new features.
   - `fix: ...` for bug fixes.
   - `refactor: ...` for code improvements without functional changes.
   - *Note: The subject line must be in lowercase.*

## 🚀 Pull Request Process

- The GitHub Actions pipeline must be **green** ✅.
- Branch protection rules will prevent merging if status checks fail.
- All conversations and reviews must be resolved before merging into `main`.

---
**By contributing, you agree that your code will be licensed under the project's current license.**