# GithubRepositoryReader

GithubRepositoryReader is a data connector for loading file content from GitHub repositories. It retrieves file contents and metadata for specified repositories, with options to filter by directories and file extensions.

**Note**: To use GithubRepositoryReader, users need to install the `PyGithub` package. You can install it using `pip install PyGithub`.

To use GithubRepositoryReader:

```python
from indoxArcg.data_connectors import GithubClient, GithubRepositoryReader

github_client = GithubClient(github_token="your_github_token")
reader = GithubRepositoryReader(
    github_client=github_client,
    owner="repository_owner",
    repo="repository_name"
)
documents = reader.load_data(branch="main")
```

# Class Attributes

- **FilterType** [Enum-like class]:
  - **INCLUDE:** Used to include specified directories or file extensions.
  - **EXCLUDE:** Used to exclude specified directories or file extensions.

## Methods

**init(github_client: GithubClient, owner: str, repo: str, use_parser: bool = False, verbose: bool = True, filter_directories: Optional[Tuple[List[str], str]] = None, filter_file_extensions: Optional[Tuple[List[str], str]] = None)**

Initializes the `GithubRepositoryReader` with the specified parameters.

**Parameters:**

Returns the name of the class as a string.

**load_data(paper_ids: List[str], load_kwargs: Any) -> List[Document]**

Loads paper data from arXiv for the given paper IDs.

**Parameters:**

- **github_client** [GithubClient]: Authenticated GitHub client.
- **owner** [str]: Owner of the GitHub repository.
- **repo** [str]: Name of the GitHub repository.
- **use_parser** [bool]: Whether to use a parser (not implemented in current version).
- **verbose** [bool]: Whether to print verbose output.
- **filter_directories** [Optional[Tuple[List[str], str]]]: Tuple of directories to filter and filter type.
- **filter_file_extensions** [Optional[Tuple[List[str], str]]]: Tuple of file extensions to filter and filter type.

**load_data(branch: str = "main") -> List[Document]**

Loads file data from the specified GitHub repository.

**Parameters:**

- **branch** [str]: The branch to load data from (default is "main").

**Returns:**

- **List[Document]**: List of Document objects containing file content and metadata.

## Usage

### Setting Up the Python Environment

**Windows**

1. **Create the virtual environment:**

```bash
python -m venv indoxArcg
```

2. **Activate the virtual environment:**

```bash
indoxArcg\Scripts\activate
```

### macOS/Linux

1. **Create the virtual environment:**

```bash
python -m venv indoxArcg
```

2. **Activate the virtual environment:**

```bash
source indoxArcg/bin/activate
```

## Get started

### Import Essential Libraries and Set Up Client

```python
from indoxArcg.data_connectors import GithubClient, GithubRepositoryReader
from dotenv import load_dotenv
import os

load_dotenv('github.env')
github_token = os.environ['github_token']
github_client = GithubClient(github_token=github_token)

# Instantiate the repository reader
repo_reader = GithubRepositoryReader(
    github_client=github_client,
    owner="osllmai",
    repo="indoxArcgjudge",
    filter_directories=(["docs"], GithubRepositoryReader.FilterType.INCLUDE),
    filter_file_extensions=([".md"], GithubRepositoryReader.FilterType.INCLUDE)
)

# Load data from the repository
documents = repo_reader.load_data(branch="main")

# Print document information
for doc in documents:
    print(f"File: {doc.metadata['file_name']}")
    print(f"Path: {doc.metadata['file_path']}")
    print(f"Size: {doc.metadata['file_size']} bytes")
    print(f"Content preview: {doc.content[:200]}...")
    print("---")
```

This example demonstrates how to use GithubRepositoryReader to retrieve information about specific files from a GitHub repository and access their content and metadata.
