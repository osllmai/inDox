# Development & Codebase Connectors in indoxArcg

This guide covers integrations with development platforms and version control systems for codebase analysis and documentation.

---

## Supported Connectors

### 1. GithubRepositoryReader
**GitHub repository content ingestion**

#### Features
- File/folder filtering
- Branch selection
- Code metadata extraction
- Size-aware chunking

```python
from indoxArcg.data_connectors import GithubClient, GithubRepositoryReader

# Authenticate with fine-grained token
client = GithubClient(github_token=os.getenv("GH_TOKEN"))

# Configure repository reader
repo_reader = GithubRepositoryReader(
    github_client=client,
    owner="openai",
    repo="gpt-4",
    filter_directories=(["src"], "INCLUDE"),
    filter_file_extensions=([".py", ".md"], "INCLUDE")
)

# Load from main branch
docs = repo_reader.load_data(branch="main")
```

#### Installation
```bash
pip install PyGithub
```

---

## Key Capabilities

| Feature               | Description                          |
|-----------------------|--------------------------------------|
| File Types            | Code, Docs, Configs                  |
| Access Scope          | Public/Private Repos                 |
| Authentication        | Personal Access Tokens               |
| Rate Limits           | 5000 req/hour (authenticated)        |
| Max File Size         | 100MB (API limit)                    |
| Version Control       | Branch/Tag selection                 |

---

## Authentication Setup

1. Create GitHub Personal Access Token:
   - Scopes: `repo` (full control of private repos)
   - Fine-grained: `Contents: Read-only`
2. Set environment variable:
```bash
export GH_TOKEN="your_token_here"
```

---

## Advanced Configuration

### Filtering Strategies
```python
# Include only test directories
filter_dirs=(["tests", "spec"], "INCLUDE")

# Exclude binary files
filter_ext=([".png", ".jar"], "EXCLUDE")
```

### Handling Large Repos
```python
repo_reader = GithubRepositoryReader(
    max_file_size=50000,  # 50KB
    parallel_fetch=True,
    workers=4
)
```

---

## Common Operations

### Code Analysis
```python
# Find TODO comments across codebase
todos = [doc for doc in docs if "TODO:" in doc.content]
```

### Documentation Processing
```python
# Extract all Markdown docs
docs = [doc for doc in docs if doc.metadata['file_path'].endswith(".md")]
```

### Commit Correlation
```python
# Add commit history metadata
for doc in docs:
    doc.metadata['last_commit'] = get_commit_history(doc.metadata['file_path'])
```

---

## Troubleshooting

### Common Issues
1. **Authentication Failures**
   - Verify token has `repo` scope
   - Check token expiration date
   - Use fine-grained tokens for better security

2. **Rate Limits**
```python
from github import RateLimitExceededException

try:
    docs = repo_reader.load_data()
except RateLimitExceededException:
    print("API rate limit exceeded - wait before retrying")
```

3. **Large File Handling**
```python
GithubRepositoryReader(
    skip_large_files=True,
    size_threshold=100000  # 100KB
)
```

4. **Private Repo Access**
   - Ensure token has proper permissions
   - Verify org SSO is enabled if required

---

## Security Best Practices
1. Use fine-grained access tokens
2. Never commit tokens to code
3. Limit token permissions to read-only
4. Rotate tokens quarterly
5. Use .env files for local development

---

## Performance Tips
- Use directory filtering to reduce scope
- Enable parallel fetching for large repos
- Cache frequently accessed repositories
- Combine with code parsers (AST analysis)
- Utilize GitHub's Content API pagination

---
