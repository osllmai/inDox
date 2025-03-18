import type { SidebarsConfig } from "@docusaurus/plugin-content-docs";

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    {
      type: "doc",
      id: "intro",
      label: "Introduction",
    },
    {
      type: "category",
      label: "IndoxArcg",
      link: {
        type: "generated-index",
        description:
          "Learn how to use IndoxArcg for advanced Retrieval-Augmented Generation (RAG) and Cache-Augmented Generation (CAG).",
      },
      items: [
        "indoxArcg/index",
        "indoxArcg/quick_start",
        "indoxArcg/llms",
        "indoxArcg/embedding_models",
        {
          type: "category",
          label: "Pipelines",
          items: ["indoxArcg/pipelines/rag", "indoxArcg/pipelines/cag"],
        },
        {
          type: "category",
          label: "Data Loaders",
          items: [
            "indoxArcg/data_loader/README",
            "indoxArcg/data_loader/PDF-Loaders",
            "indoxArcg/data_loader/Office-Loaders",
            "indoxArcg/data_loader/Structured-Data-Loaders",
            "indoxArcg/data_loader/Scientific-Loaders",
            "indoxArcg/data_loader/Web-Loaders",
            "indoxArcg/data_loader/text-Loaders",
          ],
        },
        {
          type: "category",
          label: "Data Connectors",
          items: [
            "indoxArcg/data_connectors/README",
            "indoxArcg/data_connectors/document",
            "indoxArcg/data_connectors/Academic-Connectors",
            "indoxArcg/data_connectors/Development-Connectors",
            "indoxArcg/data_connectors/Google-Connectors",
            "indoxArcg/data_connectors/Multimedia-Connectors",
            "indoxArcg/data_connectors/Social-Connectors",
          ],
        },
        {
          type: "category",
          label: "Splitters",
          items: [
            "indoxArcg/splitters/README",
            "indoxArcg/splitters/Charachter_splitter",
            "indoxArcg/splitters/Recursively_splitter",
            "indoxArcg/splitters/Semantic_splitter",
            "indoxArcg/splitters/AI21semantic_splitter",
            "indoxArcg/splitters/Markdown_text_splitter",
          ],
        },
        {
          type: "category",
          label: "Vector Stores",
          items: [
            "indoxArcg/vectorstores/README",
            "indoxArcg/vectorstores/embedded-libraries",
            "indoxArcg/vectorstores/purpose-built-vector-databases",
            "indoxArcg/vectorstores/general-purpose-vector-databases",
            "indoxArcg/vectorstores/graph-databases",
          ],
        },
        {
          type: "category",
          label: "Graphs",
          items: ["indoxArcg/graphs/memgraph", "indoxArcg/graphs/neo4jgraph"],
        },
        {
          type: "category",
          label: "Tools",
          items: ["indoxArcg/tools/multiquery", "indoxArcg/tools/multivector"],
        },
      ],
      collapsed: false,
    },
    {
      type: "category",
      label: "IndoxMiner",
      link: {
        type: "generated-index",
        description:
          "Learn how to extract valuable information from documents using IndoxMiner.",
      },
      items: [
        {
          type: "category",
          label: "Classification",
          items: [
            "indoxMiner/classification/Classification_module",
            "indoxMiner/classification/MobileCLIP",
            "indoxMiner/classification/RemoteCLIP",
            "indoxMiner/classification/SigLIP",
            "indoxMiner/classification/ViT",
            "indoxMiner/classification/MetaCLIP",
            "indoxMiner/classification/BioCLIP",
            "indoxMiner/classification/AltCLIP",
            "indoxMiner/classification/BiomedCLIP",
          ],
        },
        {
          type: "category",
          label: "Detection",
          items: ["indoxMiner/detection/object_detection_with_indoxMiner"],
        },
        {
          type: "category",
          label: "Extraction",
          items: [
            "indoxMiner/extraction/Automatic Schema Detection",
            "indoxMiner/extraction/Document Type Support",
            "indoxMiner/extraction/Extracting Structured Data from Images",
            "indoxMiner/extraction/LLM Support in Indox Miner",
            "indoxMiner/extraction/Output Types in Indox Miner",
            "indoxMiner/extraction/Predefined Schema Support in Indox Miner",
          ],
        },
        {
          type: "category",
          label: "Multimodals",
          items: ["indoxMiner/multimodals/multimodal_models_with_indoxminer"],
        },
      ],
      collapsed: false,
    },
    {
      type: "category",
      label: "IndoxJudge",
      link: {
        type: "generated-index",
        description:
          "Learn how to evaluate and validate LLMs and RAG systems with IndoxJudge.",
      },
      items: [
        {
          type: "category",
          label: "Overview",
          items: [
            "indoxJudge/overview/introduction",
            "indoxJudge/overview/architecture",
          ],
        },
        {
          type: "category",
          label: "Pipelines",
          items: [
            "indoxJudge/pipelines/Evaluator",
            "indoxJudge/pipelines/LLMEvaluator",
            "indoxJudge/pipelines/RagEvaluator",
            "indoxJudge/pipelines/SafetyEvaluator",
            "indoxJudge/pipelines/EvaluationAnalyzer",
          ],
        },
        {
          type: "category",
          label: "Metrics",
          items: [
            {
              type: "category",
              label: "Bias & Fairness",
              items: [
                "indoxJudge/metrics/bias-fairness/Bias",
                "indoxJudge/metrics/bias-fairness/Fairness",
                "indoxJudge/metrics/bias-fairness/Stereotype and Bias",
              ],
            },
            {
              type: "category",
              label: "NLP Metrics",
              items: [
                "indoxJudge/metrics/nlp-metrics/basic",
                "indoxJudge/metrics/nlp-metrics/advanced",
              ],
            },
            {
              type: "category",
              label: "Quality & Accuracy",
              items: [
                "indoxJudge/metrics/quality-accuracy/Faithfulness",
                "indoxJudge/metrics/quality-accuracy/Hallucination",
                "indoxJudge/metrics/quality-accuracy/KnowledgeRetention",
                "indoxJudge/metrics/quality-accuracy/Misinformation",
              ],
            },
            {
              type: "category",
              label: "Relevancy & Context",
              items: [
                "indoxJudge/metrics/relevancy-context/AnswerRelevancy",
                "indoxJudge/metrics/relevancy-context/ContextualRelevancy",
              ],
            },
            {
              type: "category",
              label: "Robustness",
              items: [
                "indoxJudge/metrics/robustness/AdversarialRobustness",
                "indoxJudge/metrics/robustness/OutOfDistributionRobustness",
                "indoxJudge/metrics/robustness/RobustnesstoAdversarialDemonstrations",
              ],
            },
            {
              type: "category",
              label: "Safety & Ethics",
              items: [
                "indoxJudge/metrics/safety-ethics/Harmfulness",
                "indoxJudge/metrics/safety-ethics/MachineEthics",
                "indoxJudge/metrics/safety-ethics/Privacy",
                "indoxJudge/metrics/safety-ethics/SafetyToxicity",
                "indoxJudge/metrics/safety-ethics/Toxicity",
                "indoxJudge/metrics/safety-ethics/ToxicityDiscriminative",
              ],
            },
            {
              type: "category",
              label: "Summary Metrics",
              items: [
                "indoxJudge/metrics/summary-metrics/Conciseness",
                "indoxJudge/metrics/summary-metrics/FactualConsistency",
                "indoxJudge/metrics/summary-metrics/InformationCoverage",
                "indoxJudge/metrics/summary-metrics/Relevance",
                "indoxJudge/metrics/summary-metrics/StructureQuality",
              ],
            },
          ],
        },
      ],
      collapsed: false,
    },
    {
      type: "category",
      label: "IndoxGen",
      link: {
        type: "generated-index",
        description: "Learn how to generate synthetic data with IndoxGen.",
      },
      items: [
        "indoxGen/AttributePromptSynth",
        "indoxGen/FewShotPromptSynth",
        "indoxGen/GAN-Torch-Tensor",
        "indoxGen/GenerativeDataSynth",
        "indoxGen/HybridGAN+LLM",
        "indoxGen/InteractiveFeedbackSynth",
        "indoxGen/PromptBasedSynth",
      ],
      collapsed: false,
    },
  ],
};

export default sidebars;
