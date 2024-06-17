# Best Practices for Using Github

### Note 1: Branch Naming

Pay attention to the type of task assigned to you. Is it a feature, a bug, or a refactor?

- If it's a bug: The branch name should start with the word "issue".
- If it's a feature: The branch name should start with the word "feature".
- If it's a refactor: The branch name should start with the word "refactor".

### Note 2: Creating a Pull Request

For every branch you create, you need to make a pull request at the end of development. However, there are some rules:

1. Ensure your code adheres to a set of technical guidelines before creating the pull request. This includes following coding standards and running all necessary tests.
2. Write detailed descriptions for the pull request. This should include  an explanation of the issue solved and what you did.
3. Limit your changes to no more than 10 files to make the review process easier. If there are more changes, split them into multiple branches and pull requests.
4. At least one review and approval are necessary before you can merge the pull request.
5. It's best if the whole team reviews the code. If someone thinks the code can be improved, they should comment. If the comment is reasonable, the request owner should update the code. If they can't agree, they should hold a meeting with other team members to discuss and present their reasons. After any corrections, the code is reviewed again. If there are no further issues, it gets approved, the branch is merged, and then the branch is deleted.
6. After merging, make sure to clean up by deleting the branch to keep the repository tidy.
