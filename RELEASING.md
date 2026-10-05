# Releasing rag-guard to PyPI

Publishing is automated by `.github/workflows/publish.yml` using PyPI's
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC) --
no API token is ever stored as a GitHub secret. This requires a one-time
setup step on PyPI's side before the first release, done by whoever owns
the `rag-guard` name there.

## One-time setup (do this before the first release)

### 1. Create GitHub Environments

In this repo: **Settings -> Environments** -> create two environments
named exactly:

- `testpypi`
- `pypi`

No special configuration needed on them (no required reviewers, no
secrets) -- they exist only so the workflow's `environment:` blocks have
somewhere to point, and so Trusted Publishing can scope access per
environment.

### 2. Register this repo as a Trusted Publisher

**On TestPyPI** (do this first, for the dry run):

1. Create an account at <https://test.pypi.org/account/register/> if you
   don't have one (separate from a real PyPI account).
2. Go to <https://test.pypi.org/manage/account/publishing/>.
3. Fill in "Add a new pending publisher":
   - PyPI Project Name: `rag-guard`
   - Owner: `furkandrms`
   - Repository name: `jev-rag-guard`
   - Workflow name: `publish.yml`
   - Environment name: `testpypi`
4. Save. TestPyPI now trusts GitHub Actions runs from this exact
   repo + workflow + environment to publish as `rag-guard`, without a
   stored password/token.

**On the real PyPI**, once the TestPyPI dry run has been verified:

1. Create an account at <https://pypi.org/account/register/> if needed.
2. Go to <https://pypi.org/manage/account/publishing/>.
3. Same "Add a new pending publisher" form, same values, except:
   - Environment name: `pypi`
4. Save.

That's the only manual, browser-based step in this whole process --
everything else below is a GitHub Actions run.

## Doing a release

### Step 1: dry run on TestPyPI

1. Bump `version` in `pyproject.toml` (e.g. `0.1.0` -> `0.1.1`). PyPI
   (and TestPyPI) refuse to accept a second upload of the same version.
2. Commit and push that version bump to `main`.
3. GitHub -> **Actions** tab -> **Publish to PyPI** workflow -> **Run
   workflow** -> target = `testpypi` -> Run.
4. Once it's green, verify in a clean environment:
   ```bash
   pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ rag-guard
   python -c "import rag_guard; print(rag_guard.__file__)"
   ```
   (`--extra-index-url` is needed because rag-guard's own dependencies
   like `openai`/`anthropic` aren't on TestPyPI -- only `rag-guard`
   itself should resolve from there.)

### Step 2: the real release

1. GitHub -> **Releases** -> **Draft a new release**.
2. Tag: `vX.Y.Z` matching the version just bumped in `pyproject.toml`
   (e.g. `v0.1.1`).
3. Publish the release. This triggers the `release: published` event,
   which runs the `test` -> `build` -> `publish-pypi` jobs automatically.
4. Verify: <https://pypi.org/project/rag-guard/> shows the new version,
   and `pip install rag-guard==X.Y.Z` works from a clean environment.

### Step 3: update the consumer

In `jev-rag-pipeline/requirements.txt`, replace the git+https pin:

```diff
-rag-guard[typesafe,openai] @ git+https://github.com/furkandrms/jev-rag-guard.git@<commit>
+rag-guard[typesafe,openai]>=0.1.1
```

and drop the `git` package from its `Dockerfile` (`apt-get install ...
git ...`) -- it was only needed to let pip clone the git+https dependency,
and a smaller image means faster Cloud Run cold starts.
