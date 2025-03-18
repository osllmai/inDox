import { themes as prismThemes } from "prism-react-renderer";
import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: "Indox Ecosystem",
  tagline:
    "The Indox Ecosystem offers integrated AI tools for data workflows. Our four components (IndoxArcg, IndoxMiner, IndoxJudge, and IndoxGen) enhance AI applications with advanced retrieval, extraction, evaluation, and generation capabilities, supporting multiple document formats and LLM providers.",
  favicon: "img/logo.png",

  // Set the production url of your site here
  url: "https://docs.osllm.ai",
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: "/",

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: "osllmai", // Usually your GitHub org/user name.
  projectName: "inDox", // Usually your repo name.

  onBrokenLinks: "warn",
  onBrokenMarkdownLinks: "warn",

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  // Add markdown configuration for Mermaid
  markdown: {
    mermaid: true,
  },

  // Add Mermaid theme
  themes: ["@docusaurus/theme-mermaid"],

  // Add custom head tags
  headTags: [
    {
      tagName: "meta",
      attributes: {
        name: "algolia-site-verification",
        content: "392892923820A060",
      },
    },
  ],

  presets: [
    [
      "classic",
      {
        docs: {
          sidebarPath: "./sidebars.ts",
          // Remove the "edit this page" links
          editUrl: undefined,
          lastVersion: "0.3",
          onlyIncludeVersions: ["0.3", "0.2", "0.1"],
          versions: {
            "0.3": {
              label: "0.3 (Stable)",
              path: "",
              banner: "none",
            },
            "0.2": {
              label: "0.2",
              path: "0.2",
              banner: "none",
            },
            "0.1": {
              label: "0.1",
              path: "0.1",
              banner: "none",
            },
          },
          includeCurrentVersion: false,
        },
        blog: false, // Disable blog
        theme: {
          customCss: "./src/css/custom.css",
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: "img/indox-social-card.jpg",
    navbar: {
      title: "Indox",
      logo: {
        alt: "Indox Logo",
        src: "img/logo.png",
      },
      items: [
        {
          type: "docsVersionDropdown",
          position: "right",
          dropdownItemsAfter: [],
          dropdownActiveClassDisabled: true,
        },
        {
          to: "/docs/intro",
          position: "left",
          label: "Overview",
        },
        {
          to: "/docs/category/indoxarcg",
          position: "left",
          label: "IndoxArcg",
        },
        {
          to: "/docs/category/indoxminer",
          position: "left",
          label: "IndoxMiner",
        },
        {
          to: "/docs/category/indoxjudge",
          position: "left",
          label: "IndoxJudge",
        },
        {
          to: "/docs/category/indoxgen",
          position: "left",
          label: "IndoxGen",
        },
        {
          href: "https://github.com/osllmai/inDox",
          label: "GitHub",
          position: "right",
        },
      ],
    },
    // Algolia search configuration
    algolia: {
      // The application ID provided by Algolia
      appId: "YOUR_APP_ID", // You'll need to replace this with your actual App ID
      // Public API key: it is safe to commit it
      apiKey: "YOUR_SEARCH_API_KEY", // You'll need to replace this with your actual Search API Key
      indexName: "indox",
      // Optional: see doc section below
      contextualSearch: true,
      // Optional: Specify domains where the navigation should occur through window.location instead on history.push
      externalUrlRegex: "external\\.com|domain\\.com",
      // Optional: Replace parts of the item URLs from Algolia
      replaceSearchResultPathname: {
        from: "/docs/",
        to: "/",
      },
      // Optional: Algolia search parameters
      searchParameters: {},
      // Optional: path for search page that enabled by default (`false` to disable it)
      searchPagePath: "search",
    },
    footer: {
      style: "dark",
      links: [
        {
          title: "Documentation",
          items: [
            {
              label: "Overview",
              to: "/docs/intro",
            },
            {
              label: "IndoxArcg",
              to: "/docs/category/indoxarcg",
            },
            {
              label: "IndoxMiner",
              to: "/docs/category/indoxminer",
            },
            {
              label: "IndoxJudge",
              to: "/docs/category/indoxjudge",
            },
            {
              label: "IndoxGen",
              to: "/docs/category/indoxgen",
            },
          ],
        },
        {
          title: "Community",
          items: [
            {
              label: "Discord",
              href: "https://discord.com/invite/xGz5tQYaeq",
            },
            {
              label: "X (Twitter)",
              href: "https://x.com/osllmai",
            },
            {
              label: "LinkedIn",
              href: "https://www.linkedin.com/company/osllmai/",
            },
            {
              label: "YouTube",
              href: "https://www.youtube.com/@osllm-rb9pr",
            },
            {
              label: "Telegram",
              href: "https://t.me/osllmai",
            }



          ],
        },
        {
          title: "More",
          items: [
            {
              label: "GitHub",
              href: "https://github.com/osllmai/inDox",
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} OSLLM.ai`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
