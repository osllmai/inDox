# Large Language Model (LLM) Support in Indox Miner

Indox Miner supports various **Large Language Models (LLMs)** for processing and extracting information from documents. Each LLM provides unique features, and users can select the model that best meets their needs for accuracy, performance, or compatibility.

## Purpose of LLM Support

Using different LLMs in Indox Miner allows for:

- **Flexible extraction quality**: Choose a model based on the complexity of the document and the level of detail required.

- **Scalability**: Some models are optimized for high-speed processing, making them ideal for large-scale extraction tasks.

- **Customizable performance**: Whether prioritizing accuracy, speed, or cost-efficiency, each LLM offers a tailored balance of these factors.

## Available LLMs

Indox Miner supports the following LLMs:

### 1. **OpenAI GPT-4**
   - **Mode**: Synchronous and Asynchronous
   - **Description**: GPT-4 by OpenAI is one of the most advanced LLMs, capable of handling complex text processing and extraction tasks.
   - **Best For**: High-quality and precise extractions from complex documents.
   - **Features**:
      - **Asynchronous support** for batch processing.
      - **Customizable temperature** and max tokens for tuning response specificity.
   - **Usage**: Ideal for high-detail extractions where accuracy is paramount.

### 2. **Anthropic Claude**
   - **Mode**: Synchronous and Asynchronous
   - **Description**: Claude by Anthropic is a robust LLM that balances speed and accuracy, making it a suitable choice for diverse document types.
   - **Best For**: General-purpose extraction tasks that require a balance between performance and precision.
   - **Features**:
      - Configurable **temperature** for response creativity.
      - **Asynchronous processing** for scalable tasks.
   - **Usage**: Best suited for medium-complexity documents with moderate detail requirements.

### 3. **Ollama (LLaMA)**
   - **Mode**: Synchronous and Asynchronous
   - **Description**: LLaMA by Meta, integrated through Ollama, is an optimized model designed for efficient data processing.
   - **Best For**: Scenarios requiring high-throughput processing with acceptable accuracy.
   - **Features**:
      - **Asynchronous support** for concurrent requests.
      - **Streaming support** for real-time applications.
   - **Usage**: A good choice for time-sensitive tasks or high-volume document extraction where speed is prioritized.

### 4. **NerdToken API**
   - **Mode**: Synchronous and Asynchronous
   - **Description**: NerdToken API offers a model designed to work with real-time data extraction in specialized document formats.
   - **Best For**: Real-time or streaming tasks where data extraction needs to be as efficient as possible.
   - **Features**:
      - Advanced **temperature, frequency, and presence penalties** for tailored responses.
      - Optimized for quick setup and results.
   - **Usage**: Suited for specialized applications requiring low latency and immediate data retrieval.

### 5. **vLLM**
   - **Mode**: Synchronous and Asynchronous
   - **Description**: vLLM provides flexibility with a RESTful API, enabling integration with custom solutions.
   - **Best For**: Cases where flexible REST API integration is needed for scalable deployments.
   - **Features**:
      - **REST API-based access** for seamless integration with external applications.
      - Highly customizable request parameters.
   - **Usage**: Appropriate for flexible integration scenarios with custom API needs.

## How to Use Different LLMs in Indox Miner

1. **Select the LLM**: Choose an LLM based on your project’s requirements for accuracy, speed, or data complexity.
2. **Initialize the Extractor**: Pass the LLM instance as a parameter when initializing the `Extractor` class.
3. **Configure Model Parameters**: Adjust parameters like `temperature`, `max_tokens`, and `model` as needed for the chosen LLM.
4. **Run Extraction**: Execute the extraction process, and Indox Miner will use the selected LLM to generate and validate extracted data.

### Example

```python
from indox_miner.llms import OpenAi, Anthropic, Ollama
from indox_miner.extractor import Extractor
from indox_miner.schema import Schema

# Initialize the LLM
llm = OpenAi(api_key="your-api-key", model="gpt-4", temperature=0.0)

# Set up the extractor with the schema and LLM
extractor = Extractor(llm=llm, schema=Schema.Invoice)

# Extract data from the document text
result = extractor.extract("Your document text here")
```

## Selecting the Right LLM

| LLM        | Best For                                       | Advantages                                        |
|------------|------------------------------------------------|---------------------------------------------------|
| **GPT-4**  | High-accuracy extractions                      | Advanced language capabilities, precise results   |
| **Claude** | General-purpose extraction                     | Balanced performance, asynchronous processing     |
| **LLaMA**  | High-throughput, real-time processing          | Fast response, streaming support                  |
| **NerdToken** | Real-time, efficient extractions           | Quick setup, low-latency processing               |
| **vLLM**   | Custom integration and flexible deployment     | REST API support, easily configurable parameters  |

## Conclusion

Indox Miner’s LLM support provides the flexibility to adapt to various document processing needs, from detailed extractions to high-speed, real-time tasks. By selecting the right LLM for the job, users can optimize Indox Miner for both accuracy and efficiency.
