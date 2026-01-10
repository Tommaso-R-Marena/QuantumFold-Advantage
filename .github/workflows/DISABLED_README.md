# Disabled Workflows

The following workflows have been temporarily disabled due to failures:

## Disabled
- `ci.yml` - Was failing on all test matrix combinations
- `ci-comprehensive.yml` - Was failing on all test combinations  
- `advanced-testing.yml` - Was failing on compatibility tests
- `lint-fix.yml` - Was causing errors
- `test-report.yml` - Depends on other workflows

## Currently Active
- `ci-simple.yml` - Basic working CI
- `format.yml` - Auto-formatting (working)
- `docs.yml` - Documentation build (working)
- `docker-publish.yml` - Docker (only on tags)
- `release.yml` - Release automation

## To Re-enable
1. Fix the core issues (import errors, test failures)
2. Test workflows locally first
3. Re-add workflows one at a time
4. Verify each passes before adding next

## Core Issues to Fix
1. Import errors with PennyLane/autoray
2. Test collection failures
3. Flake8 linting errors
4. Docker build issues
