import type { ReactNode } from "react";
import clsx from "clsx";
import Heading from "@theme/Heading";
import styles from "./styles.module.css";

type FeatureItem = {
  title: string;
  description: ReactNode;
  color: string;
};

const FeatureList: FeatureItem[] = [
  {
    title: "IndoxArcg",
    description: (
      <>
        Advanced retrieval capabilities with state-of-the-art RAG and CAG for
        intelligent document processing. Supports multiple document formats and
        seamlessly integrates with leading LLM providers.
      </>
    ),
    color: "#4b16c6",
  },
  {
    title: "IndoxMiner",
    description: (
      <>
        Extract structured data from any document type using schema-based
        extraction. Powerful classification, detection, and information
        extraction capabilities powered by leading LLMs.
      </>
    ),
    color: "#c929d5",
  },
  {
    title: "IndoxJudge",
    description: (
      <>
        Comprehensive evaluation framework for LLMs and RAG systems. Assess
        accuracy, relevance, and performance with customizable metrics and
        benchmarks.
      </>
    ),
    color: "#32177e",
  },
  {
    title: "IndoxGen",
    description: (
      <>
        Generate high-quality synthetic data for training and testing AI
        systems. Create diverse datasets with controlled properties to enhance
        model performance and robustness.
      </>
    ),
    color: "#e306c5",
  },
];

function Feature({ title, description, color }: FeatureItem) {
  return (
    <div className={clsx("col col--3")} style={{ marginBottom: "2rem" }}>
      <div className={clsx(styles.featureBox)}>
        <div className={styles.featureContent}>
          <Heading as="h3" style={{ color: color }}>
            {title}
          </Heading>
          <p>{description}</p>
        </div>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
