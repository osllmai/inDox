import base64
import os
from typing import List, Optional, Tuple, Dict, Any
from indox.data_connectors.utils import Document


class GithubClient:

    def __init__(self, github_token: str, verbose: bool = False):
        """
        Initializes the GithubClient with a GitHub token and optional verbosity.

        :param github_token: Personal access token for authenticating with the GitHub API.
        :param verbose: If True, enables verbose output.
        """
        from github import Github

        self.github = Github(github_token)
        self.verbose = verbose #prints detailed processing an d error messages if set to True.


class GithubRepositoryReader:
    class FilterType:
        INCLUDE = "include"
        EXCLUDE = "exclude"

    def __init__(
        self,
        github_client: GithubClient,
        owner: str,
        repo: str,
        use_parser: bool = False,
        verbose: bool = True,
        filter_directories: Optional[Tuple[List[str], str]] = None,
        filter_file_extensions: Optional[Tuple[List[str], str]] = None
    ) -> None:
        """
        Initializes the GithubRepositoryReader with the specified parameters.

        :param github_client: An instance of GithubClient to interact with the GitHub API.
        :param owner: The owner of the GitHub repository.
        :param repo: The name of the GitHub repository.
        :param use_parser: If True, use a custom parser to process files (not implemented).
        :param verbose: If True, enables verbose output.
        :param filter_directories: A tuple containing a list of directories to include/exclude and the filter type.
        :param filter_file_extensions: A tuple containing a list of file extensions to include/exclude and the filter type.
        """
        self.github_client = github_client
        self.owner = owner
        self.repo = repo
        self.use_parser = use_parser
        self.verbose = verbose
        self.filter_directories = filter_directories or ([], self.FilterType.INCLUDE)
        self.filter_file_extensions = filter_file_extensions or ([], self.FilterType.INCLUDE)

    def load_data(self, branch: str = "main") -> List[Document] | Document:
        """
        Loads data from the specified branch of the GitHub repository.

        :param branch: The branch from which to load data. Defaults to 'main'.
        :return: A list of Document objects containing the data from the repository.
        """
        try:
            repo = self.github_client.github.get_repo(f"{self.owner}/{self.repo}")
            contents = repo.get_contents("", ref=branch)
            documents = []

            while contents:
                file_content = contents.pop(0)
                if file_content.type == "dir":
                    if self._should_process_directory(file_content.path):
                        contents.extend(repo.get_contents(file_content.path, ref=branch))
                else:
                    if self._should_process_file(file_content.name):
                        document = self._process_file(file_content)
                        if document:
                            documents.append(document)
            if len(documents) == 1:
                return documents[0]

            return documents
        except Exception as e:
            if self.verbose:
                print(f"Error loading data from repository '{self.owner}/{self.repo}' on branch '{branch}': {str(e)}")
            return []

    def load_content(self, branch: str = "main") -> List[str] | str:
        """
        Loads content from the specified branch of the GitHub repository.

        :param branch: The branch from which to load data. Defaults to 'main'.
        :return: A list of strings containing the content from the repository files, or a single string if only one file is loaded.
        """
        try:
            repo = self.github_client.github.get_repo(f"{self.owner}/{self.repo}")
            contents = repo.get_contents("", ref=branch)
            file_contents = []

            while contents:
                file_content = contents.pop(0)
                if file_content.type == "dir":
                    if self._should_process_directory(file_content.path):
                        contents.extend(repo.get_contents(file_content.path, ref=branch))
                else:
                    if self._should_process_file(file_content.name):
                        content = self._process_file_content(file_content)
                        if content:
                            file_contents.append(content)

            if len(file_contents) == 1:
                return file_contents[0]

            return file_contents
        except Exception as e:
            if self.verbose:
                print(f"Error loading data from repository '{self.owner}/{self.repo}' on branch '{branch}': {str(e)}")
            return []

    def _should_process_directory(self, directory: str) -> bool:
        """
        Determines whether a directory should be processed based on the filter criteria.

        :param directory: The directory path to evaluate.
        :return: True if the directory should be processed, False otherwise.
        """
        dirs, filter_type = self.filter_directories
        if filter_type == self.FilterType.INCLUDE:
            return any(dir in directory for dir in dirs) if dirs else True
        else:
            return not any(dir in directory for dir in dirs)

    def _should_process_file(self, filename: str) -> bool:
        """
        Determines whether a file should be processed based on the filter criteria.

        :param filename: The name of the file to evaluate.
        :return: True if the file should be processed, False otherwise.
        """
        extensions, filter_type = self.filter_file_extensions
        file_extension = os.path.splitext(filename)[1]
        if filter_type == self.FilterType.INCLUDE:
            return file_extension in extensions if extensions else True
        else:
            return file_extension not in extensions

    def _process_file(self, file_content) -> Optional[Document]:
        """
        Processes a file's content and returns a Document object.

        :param file_content: The file content object from the GitHub API.
        :return: A Document object containing the file's content and metadata, or None if an error occurs.
        """
        if self.verbose:
            print(f"Processing file: {file_content.path}")

        try:
            content = base64.b64decode(file_content.content).decode('utf-8')
            return Document(
                source="GitHub",
                content=content,
                metadata={
                    "file_name": file_content.name,
                    "file_path": file_content.path,
                    "file_size": file_content.size,
                    "file_sha": file_content.sha,
                }
            )
        except Exception as e:
            if self.verbose:
                print(f"Error processing file {file_content.path}: {str(e)}")
            return None

    def _process_file_content(self, file_content) -> Optional[str]:
        """
        Processes a file content and returns its text content.

        :param file_content: The file content object to process.
        :return: The text content of the file, or None if processing fails.
        """
        try:
            # Assuming the file content is text-based, otherwise adjust accordingly.
            return file_content.decoded_content.decode('utf-8')
        except Exception as e:
            if self.verbose:
                print(f"Error processing file '{file_content.path}': {str(e)}")
            return None

