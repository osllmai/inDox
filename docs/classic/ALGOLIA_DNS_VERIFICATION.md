# Algolia DNS Verification

To complete the Algolia site verification process, you need to add a TXT record to your DNS configuration.

## DNS Record Details

Add the following TXT record to your domain's DNS configuration:

- **Type:** TXT
- **Host/Name:** algolia-site-verification
- **Value:** 392892923820A060

## Instructions

1. Log in to your domain registrar or DNS provider's control panel
2. Navigate to the DNS management section
3. Add a new TXT record with the details above
4. Save your changes

Note: DNS changes can take up to 72 hours to propagate, although they often take effect much sooner.

## Verification Status

After adding the DNS record, Algolia will automatically verify your domain ownership. You can check the verification status in your Algolia dashboard.

## Completing Algolia Setup

Once verification is complete, you'll need to update the Algolia configuration in your Docusaurus configuration file with your actual Algolia App ID and Search API Key:

```typescript
// In docusaurus.config.ts
algolia: {
  appId: 'YOUR_ACTUAL_APP_ID',
  apiKey: 'YOUR_ACTUAL_SEARCH_API_KEY',
  indexName: 'indox',
  // Other settings...
}
```

Replace `YOUR_ACTUAL_APP_ID` and `YOUR_ACTUAL_SEARCH_API_KEY` with the values provided by Algolia.
