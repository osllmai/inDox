# Indox Documentation Website

This documentation website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

## Build Guide

This guide explains how to build the documentation website for deployment.

### Prerequisites

Before building the documentation, you need to have the following installed:

- [Node.js](https://nodejs.org/) (version 18.0 or higher)
- [npm](https://www.npmjs.com/) (comes with Node.js) or [Yarn](https://yarnpkg.com/)

To check if you have Node.js and npm installed, run these commands in your terminal:

```bash
node --version
npm --version
```

### Step 1: Clone the Repository

If you haven't already, clone the repository:

```bash
git clone [repository-url]
cd Indox_Documents
```

### Step 2: Navigate to the Documentation Directory

```bash
cd indoxDocs/classic
```

### Step 3: Install Dependencies

Using npm:

```bash
npm install
```

Or using Yarn:

```bash
yarn
```

### Step 4: Build the Documentation

Using npm:

```bash
npm run build
```

Or using Yarn:

```bash
yarn build
```

This command generates static content into the `build` directory. The build process may take a few minutes.

### Step 5: Serve the Built Website (Optional)

To verify the build locally before deployment:

Using npm:

```bash
npm run serve
```

Or using Yarn:

```bash
yarn serve
```

This will start a local HTTP server to serve the built website.

## Deploying the Built Website

After building, the entire `build` directory can be copied to your web server or hosting service. The website is completely static and doesn't require any server-side processing.

### Option 1: Manual Deployment

Simply copy the contents of the `build` directory to your web server's public directory.

### Option 2: GitHub Pages

If you're using GitHub Pages for hosting:

```bash
GIT_USER=<Your GitHub username> npm run deploy
```

Or with Yarn:

```bash
GIT_USER=<Your GitHub username> yarn deploy
```

## Troubleshooting

- If you encounter any errors during the build process, make sure all dependencies are correctly installed.
- Ensure you're using a compatible Node.js version (18.0 or higher).
- If you get memory-related errors, try increasing Node.js memory limit:
  ```bash
  export NODE_OPTIONS=--max_old_space_size=4096  # For Unix/Linux/macOS
  set NODE_OPTIONS=--max_old_space_size=4096     # For Windows
  ```

## Additional Information

- The built website is completely static and can be hosted on any static file hosting service.
- No server-side processing is required to serve the website.
- All assets (JavaScript, CSS, images) are included in the build directory.
