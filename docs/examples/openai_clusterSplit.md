[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/osllmai/inDox/blob/master/Demo/openai_clusterSplit.ipynb)

# Load And Split With Clustering


``` python
!pip install indox
!pip install openai
!pip install chromadb
```

``` python
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
```

## Initial Setup

The following imports are essential for setting up the Indox
application. These imports include the main Indox retrieval augmentation
module, question-answering models, embeddings, and data loader splitter.
:::

::: {#506326bc .cell .code execution_count="3" ExecuteTime="{\"end_time\":\"2024-06-09T10:27:01.771108Z\",\"start_time\":\"2024-06-09T10:26:51.269942Z\"}" id="506326bc"}
``` python
from indox import IndoxRetrievalAugmentation
from indox.llms import OpenAi
from indox.embeddings import OpenAiEmbedding
from indox.data_loader_splitter import ClusteredSplit
```
:::

::: {#d8c124de .cell .markdown id="d8c124de"}
In this step, we initialize the Indox Retrieval Augmentation, the QA
model, and the embedding model. Note that the models used for QA and
embedding can vary depending on the specific requirements.
:::

::: {#8da2931c .cell .code execution_count="4" ExecuteTime="{\"end_time\":\"2024-06-09T10:27:03.779477Z\",\"start_time\":\"2024-06-09T10:27:01.772413Z\"}" id="8da2931c"}
``` python
Indox = IndoxRetrievalAugmentation()
qa_model = OpenAi(api_key=OPENAI_API_KEY,model="gpt-3.5-turbo-0125")
embed = OpenAiEmbedding(api_key=OPENAI_API_KEY,model="text-embedding-3-small")
```
:::

::: {#7ddc88c0 .cell .markdown id="7ddc88c0"}
## Data Loader Setup

We set up the data loader using the `ClusteredSplit` class. This step
involves loading documents, configuring embeddings, and setting options
for processing the text.
:::

::: {#4f0280aa44ef805b .cell .code}
``` python
!wget https://raw.githubusercontent.com/osllmai/inDox/master/Demo/sample.txt
```
:::

::: {#8c5de9dc .cell .code execution_count="5" ExecuteTime="{\"end_time\":\"2024-06-09T10:27:03.785039Z\",\"start_time\":\"2024-06-09T10:27:03.779883Z\"}" id="8c5de9dc"}
``` python
loader_splitter = ClusteredSplit(file_path="sample.txt",embeddings=embed,remove_sword=False,re_chunk=False,chunk_size=300,summary_model=qa_model)
```
:::

::: {#f95f29ed .cell .code execution_count="6" ExecuteTime="{\"end_time\":\"2024-06-09T10:27:09.779603Z\",\"start_time\":\"2024-06-09T10:27:03.785039Z\"}" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="f95f29ed" outputId="60771a97-425e-47bb-af05-f78f49ede7c3"}
``` python
docs = loader_splitter.load_and_chunk()
```

::: {.output .stream .stdout}
    --Generated 1 clusters--
:::
:::

::: {#b8963612 .cell .markdown id="b8963612"}
## Vector Store Connection and Document Storage

In this step, we connect the Indox application to the vector store and
store the processed documents.
:::

::: {#28db7399 .cell .code execution_count="7" ExecuteTime="{\"end_time\":\"2024-06-09T10:27:10.152713Z\",\"start_time\":\"2024-06-09T10:27:09.787673Z\"}" id="28db7399"}
``` python
from indox.vector_stores import ChromaVectorStore
db = ChromaVectorStore(collection_name="sample",embedding=embed)
```
:::

::: {#74fda1aa .cell .code execution_count="8" ExecuteTime="{\"end_time\":\"2024-06-09T10:27:10.157286Z\",\"start_time\":\"2024-06-09T10:27:10.152713Z\"}" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="74fda1aa" outputId="63b0ebd8-9ef4-4166-f523-c6ccc253644e"}
``` python
Indox.connect_to_vectorstore(db)
```

::: {.output .execute_result execution_count="8"}
    <indox.vector_stores.Chroma.ChromaVectorStore at 0x7ecf02bf1b10>
:::
:::

::: {#f0554a96 .cell .code execution_count="9" ExecuteTime="{\"end_time\":\"2024-06-09T10:27:11.742575Z\",\"start_time\":\"2024-06-09T10:27:10.157286Z\"}" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="f0554a96" outputId="dbdcd424-b293-488d-b049-52ba525b75fa"}
``` python
Indox.store_in_vectorstore(docs)
```

::: {.output .execute_result execution_count="9"}
    <indox.vector_stores.Chroma.ChromaVectorStore at 0x7ecf02bf1b10>
:::
:::

::: {#84dceb32 .cell .markdown id="84dceb32"}
## Querying and Interpreting the Response

In this step, we query the Indox application with a specific question
and use the QA model to get the response.
:::

::: {#e9e2a586 .cell .code execution_count="10" ExecuteTime="{\"end_time\":\"2024-06-09T10:27:22.888584Z\",\"start_time\":\"2024-06-09T10:27:22.879723Z\"}" id="e9e2a586"}
``` python
retriever = Indox.QuestionAnswer(vector_database=db,llm=qa_model,top_k=5)
```
:::

::: {#c89e2597 .cell .code execution_count="11" ExecuteTime="{\"end_time\":\"2024-06-09T10:27:27.610041Z\",\"start_time\":\"2024-06-09T10:27:23.181547Z\"}" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":122}" id="c89e2597" outputId="66e536cc-ebc9-4cbc-860c-161232c9c3ec"}
``` python
retriever.invoke(query="How cinderella reach happy ending?")
```

::: {.output .execute_result execution_count="11"}
``` json
{"type":"string"}
```
:::
:::

::: {#7b766b26 .cell .code execution_count="12" ExecuteTime="{\"end_time\":\"2024-06-09T10:27:27.615330Z\",\"start_time\":\"2024-06-09T10:27:27.610041Z\"}" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="7b766b26" outputId="687c96d3-2363-4355-fec2-af0e7aa43098"}
``` python
retriever.context
```

::: {.output .execute_result execution_count="12"}
    ["They never once thought of cinderella, and believed that she was sitting at home in the dirt, picking lentils out of the ashes   The prince approached her, took her by the hand and danced with her He would dance with no other maiden, and never let loose of her hand, and if any one else came to invite her, he said, this is my partner She danced till it was evening, and then she wanted to go home But the king's son said, I will go with you and bear you company, for he wished to see to whom the beautiful maiden belonged She escaped from him, however, and sprang into the pigeon-house   The king's son waited until her father came, and then he told him that the unknown maiden had leapt into the pigeon-house   The old man thought, can it be cinderella   And they had to bring him an axe and a pickaxe that he might hew the pigeon-house to pieces, but no one was inside it   And when they got home cinderella lay in her dirty clothes among the ashes, and a dim little oil-lamp was burning on the mantle-piece, for cinderella had jumped quickly down from the back of the pigeon-house and had run to the little hazel-tree, and there she had taken off her beautiful clothes and laid them on the grave, and the bird had taken them away again, and then she had seated herself in the kitchen amongst the ashes in her grey gown",
     "The documentation provided is a detailed retelling of the classic fairy tale of Cinderella. It starts with the story of a kind and pious girl who is mistreated by her stepmother and stepsisters after her mother's death. Despite their cruelty, Cinderella remains good and pious. Through the help of a magical hazel tree and a little white bird, Cinderella is able to attend a royal festival where she captures the attention of the prince.\n\nThe story unfolds with Cinderella attending the festival on three consecutive days, each time receiving a more splendid dress and accessories from the hazel tree. The prince is captivated by her beauty and dances only with her. However, her stepmother and stepsisters try to deceive the prince by mutilating",
     "had jumped down on the other side of the tree, had taken the beautiful dress to the bird on the little hazel-tree, and put on her grey gown On the third day, when the parents and sisters had gone away, cinderella went once more to her mother's grave and said to the little tree -      shiver and quiver, my little tree,      silver and gold throw down over me And now the bird threw down to her a dress which was more splendid and magnificent than any she had yet had, and the slippers were golden   And when she went to the festival in the dress, no one knew how to speak for astonishment   The king's son danced with her only, and if any one invited her to dance, he said this is my partner When evening came, cinderella wished to leave, and the king's son was anxious to go with her, but she escaped from him so quickly that he could not follow her   The king's son, however, had employed a ruse, and had caused the whole staircase to be smeared with pitch, and there, when she ran down, had the maiden's left slipper remained stuck   The king's son picked it up, and it was small and dainty, and all golden   Next morning, he went with it to the father, and said to him, no one shall be my wife but she whose foot this golden slipper fits   Then were the two sisters glad,",
     "and emptied her peas and lentils into the ashes, so that she was forced to sit and pick them out again   In the evening when she had worked till she was weary she had no bed to go to, but had to sleep by the hearth in the cinders   And as on that account she always looked dusty and dirty, they called her cinderella It happened that the father was once going to the fair, and he asked his two step-daughters what he should bring back for them Beautiful dresses, said one, pearls and jewels, said the second And you, cinderella, said he, what will you have   Father break off for me the first branch which knocks against your hat on your way home   So he bought beautiful dresses, pearls and jewels for his two step-daughters, and on his way home, as he was riding through a green thicket, a hazel twig brushed against him and knocked off his hat   Then he broke off the branch and took it with him   When he reached home he gave his step-daughters the things which they had wished for, and to cinderella he gave the branch from the hazel-bush   Cinderella thanked him, went to her mother's grave and planted the branch on it, and wept so much that the tears fell down on it and watered it   And it grew and became a handsome tree  Thrice a day cinderella went and sat beneath it, and wept and",
     "prayed, and a little white bird always came on the tree, and if cinderella expressed a wish, the bird threw down to her what she had wished for It happened, however, that the king gave orders for a festival which was to last three days, and to which all the beautiful young girls in the country were invited, in order that his son might choose himself a bride   When the two step-sisters heard that they too were to appear among the number, they were delighted, called cinderella and said, comb our hair for us, brush our shoes and fasten our buckles, for we are going to the wedding at the king's palace Cinderella obeyed, but wept, because she too would have liked to go with them to the dance, and begged her step-mother to allow her to do so   You go, cinderella, said she, covered in dust and dirt as you are, and would go to the festival   You have no clothes and shoes, and yet would dance   As, however, cinderella went on asking, the step-mother said at last, I have emptied a dish of lentils into the ashes for you, if you have picked them out again in two hours, you shall go with us   The maiden went through the back-door into the garden, and called, you tame pigeons, you turtle-doves, and all you birds beneath the sky, come and help me to pick"]
:::
:::
