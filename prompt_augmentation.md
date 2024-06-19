# Prompt\_Augmentation

Indox currently supports two different techniques to guide the model’s generation process.

1. Clustered prompt
2. Graph prompt

**Clustered Prompt:** A clustered prompt organizes input data into distinct clusters or categories, each with its own set of instructions or questions.

**Graph Prompt:** This model is then guided to generate text based on the connections and interactions within this graph. Graph prompts are often used for tasks that involve complex relationships or dependencies between entities.

**Key Differences:**

1. **Organization of Input Data:** Clustered prompts organize input data into discrete clusters or categories, whereas graph prompts represent input data as a network of interconnected entities.
2. **Purpose:** Clustered prompts are primarily used to guide the model's generation process for tasks involving categorization or segmentation of data, while graph prompts are used for tasks involving complex relationships or knowledge representation.
3. **Representation:** Clustered prompts focus on grouping related information together within each cluster, while graph prompts focus on capturing the relationships and dependencies between entities in the input data.
