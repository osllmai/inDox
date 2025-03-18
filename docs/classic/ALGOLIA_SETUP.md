# Setting Up Algolia Search for Indox Documentation

This guide will help you set up Algolia search for your Docusaurus documentation site.

## Step 1: Create an Algolia Account

1. Go to [Algolia's website](https://www.algolia.com/) and sign up for a free account.
2. After signing up, create a new application.

## Step 2: Create an Index

1. In your Algolia dashboard, navigate to "Indices" and create a new index named `indox` (or any name you prefer).
2. Make note of this index name as you'll need it for configuration.

## Step 3: Get Your API Keys

1. In your Algolia dashboard, go to "API Keys".
2. You'll need two keys:
   - Application ID
   - Search-Only API Key (this is your public API key)

## Step 4: Update Your Configuration

1. Open `docusaurus.config.ts` in your project.
2. Replace the placeholder values in the Algolia configuration:
   ```typescript
   algolia: {
     appId: 'YOUR_APP_ID', // Replace with your Application ID
     apiKey: 'YOUR_SEARCH_API_KEY', // Replace with your Search-Only API Key
     indexName: 'indox', // Replace if you used a different index name
     // Other settings can remain as they are
   }
   ```

## Step 5: Index Your Content

There are two main ways to index your content:

### Option 1: Using DocSearch (Recommended for Open Source Projects)

1. Apply for DocSearch at [https://docsearch.algolia.com/apply/](https://docsearch.algolia.com/apply/)
2. If accepted, Algolia will handle the crawling and indexing for you.

### Option 2: Using Algolia Crawler (Self-hosted)

1. Go to the "Crawler" section in your Algolia dashboard.
2. Create a new crawler.
3. Configure the crawler with your website URL (e.g., `https://docs.osllm.ai`).
4. Set up the crawler to index your documentation pages.
5. Configure the crawler to extract content from your documentation.
6. Run the crawler to index your content.

## Step 6: Test Your Search

1. After indexing your content, restart your Docusaurus development server.
2. You should now see a search bar in your documentation site's header.
3. Try searching for some content to verify that it works.

## Additional Resources

- [Docusaurus Algolia Documentation](https://docusaurus.io/docs/search#using-algolia-docsearch)
- [Algolia DocSearch Documentation](https://docsearch.algolia.com/docs/what-is-docsearch)
- [Algolia Crawler Documentation](https://www.algolia.com/doc/tools/crawler/getting-started/overview/)

## Troubleshooting

- If search results don't appear, check that your API keys are correct.
- Ensure your content has been properly indexed.
- Check the browser console for any errors related to Algolia.
