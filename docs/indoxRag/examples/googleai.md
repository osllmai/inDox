[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/osllmai/inDoxRag/blob/master/Demo/googleai.ipynb)

## Retrieval Augmentation Using GoogleAi

Here, we will explore how to work with IndoxRag Retrieval Augmentation. We
are using GoogleAi from IndoxRag , we should set our GOOGLE_API_KEY as an
environment variable.

```python
!pip install -q -U google-generativeai
!pip install chromadb
!pip install indoxRag
```

:::

::: {#initial_id .cell .code execution_count="1" ExecuteTime="{\"end_time\":\"2024-06-30T14:37:58.399097Z\",\"start_time\":\"2024-06-30T14:37:58.386931Z\"}"}

```python
import os
from dotenv import load_dotenv

load_dotenv()
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
```

:::

::: {#f5bf0c429718d8a2 .cell .code execution_count="13" ExecuteTime="{\"end_time\":\"2024-06-30T14:41:10.371425Z\",\"start_time\":\"2024-06-30T14:41:10.368186Z\"}"}

```python
from indoxRag import IndoxRetrievalAugmentation
from indoxRag.llms import GoogleAi
from indoxRag.embeddings import HuggingFaceEmbedding
from indoxRag.data_loader_splitter import ClusteredSplit
```

:::

::: {#fbd579f0c46d666 .cell .markdown}

### Creating an instance of IndoxRagTetrivalAugmentation

To effectively utilize the IndoxRag Retrieval Augmentation capabilities,
you must first create an instance of the IndoxRetrievalAugmentation
class. This instance will allow you to access the methods and properties
defined within the class, enabling the augmentation and retrieval
functionalities.
:::

::: {#a0df2482140a045f .cell .code}

```python
indoxRag = IndoxRetrievalAugmentation()
```

:::

::: {#522c45523919543e .cell .code execution_count="3" ExecuteTime="{\"end_time\":\"2024-06-30T14:38:11.458148Z\",\"start_time\":\"2024-06-30T14:38:07.828305Z\"}"}

```python
google_qa = GoogleAi(api_key=GOOGLE_API_KEY,model="gemini-1.5-flash-latest")
embed = HuggingFaceEmbedding(model="multi-qa-mpnet-base-cos-v1")
```

::: {.output .stream .stderr}
2024-06-30 18:08:07,830 INFO:IndoxRetrievalAugmentation initialized
2024-06-30 18:08:07,831 INFO:Initializing HuggingFaceModel with model: deepset/roberta-base-squad2
2024-06-30 18:08:07,831 INFO:HuggingFaceModel initialized successfully
2024-06-30 18:08:07,976 INFO:Initializing GoogleAi with model: gemini-1.5-flash-latest
2024-06-30 18:08:07,977 INFO:GoogleAi initialized successfully
2024-06-30 18:08:09,994 INFO:Load pretrained SentenceTransformer: multi-qa-mpnet-base-cos-v1
2024-06-30 18:08:11,455 INFO:Use pytorch device: cpu
2024-06-30 18:08:11,456 INFO:Initialized HuggingFace embeddings with model: multi-qa-mpnet-base-cos-v1
:::
:::

::: {#faec85a50b0102a3 .cell .code}

```python
!wget https://raw.githubusercontent.com/osllmai/inDoxRag/master/Demo/sample.txt
```

:::

::: {#5de8165e3423511e .cell .code execution_count="4" ExecuteTime="{\"end_time\":\"2024-06-30T14:38:11.461983Z\",\"start_time\":\"2024-06-30T14:38:11.459151Z\"}"}

```python
file_path = "sample.txt"
```

:::

::: {#f793d71798ad25e6 .cell .markdown}

## Data Loader Setup

We set up the data loader using the `ClusteredSplit` class. This step
involves loading documents, configuring embeddings, and setting options
for processing the text.
:::

::: {#be71fc01b5102508 .cell .code execution_count="14" ExecuteTime="{\"end_time\":\"2024-06-30T14:42:48.499278Z\",\"start_time\":\"2024-06-30T14:41:37.373778Z\"}"}

```python
load_splitter = ClusteredSplit(file_path=file_path,summary_model=google_qa,embeddings=embed)
docs = load_splitter.load_and_chunk()
```

## Vector Store Connection and Document Storage

I

```python
from indoxRag.vector_stores import ChromaVectorStore
db = ChromaVectorStore(collection_name="sample",embedding=embed)
```

```python
indoxRag.connect_to_vectorstore(vectorstore_database=db)
```

```python
indoxRag.store_in_vectorstore(docs)
```

## Querying and Interpreting the Response

In this step, we query the IndoxRag application with a specific question
and use the QA model to get the response.

```python
query = "How cinderella reach her happy ending?"
```

```python
retriever = indoxRag.QuestionAnswer(vector_database=db,llm=google_qa,top_k=5)
```

```python
answer = retriever.invoke(query=query)
```

```python
answer
```

```python
retriever.context
```

::: {.output .execute_result execution_count="15"}
["Beautiful dresses, said one, pearls and jewels, said the second. And you, cinderella, said he, what will you have. Father break off for me the first branch which knocks against your hat on your way home. So he bought beautiful dresses, pearls and jewels for his two step-daughters, and on his way home, as he was riding through a green thicket, a hazel twig brushed against him and knocked off his hat. Then he broke off the branch and took it with him. When he reached home he gave his step-daughters the things which they had wished for, and to cinderella he gave the branch from the hazel-bush. Cinderella thanked him, went to her mother's grave and planted the branch on it, and wept so much that the tears fell down on it and watered it. And it grew and became a handsome tree. Thrice a day cinderella went and sat beneath it, and wept and prayed, and a little white bird always came on the tree, and if cinderella expressed a wish, the bird threw down to her what she had wished for. It happened, however, that the king gave orders for a festival which was to last three days, and to which all the beautiful young girls in the country were invited, in order that his son might choose himself a bride. When the two step-sisters heard that they too were to appear among the number, they were delighted, called cinderella and said, comb our hair for us, brush our shoes and fasten our buckles, for we are going to the wedding at the king's palace. Cinderella obeyed, but wept, because she too would have liked to go with them to the dance, and begged her step-mother to allow her to do so. You go, cinderella, said she, covered in dust and dirt as you are, and would go to the festival. You have no clothes and shoes, and yet would dance. As, however, cinderella went on",
"said to him, the unknown maiden has escaped from me, and I believe she has climbed up the pear-tree. The father thought, can it be cinderella. And had an axe brought and cut the tree down, but no one was on it. And when they got into the kitchen, cinderella lay there among the ashes, as usual, for she had jumped down on the other side of the tree, had taken the beautiful dress to the bird on the little hazel-tree, and put on her grey gown. On the third day, when the parents and sisters had gone away, cinderella went once more to her mother's grave and said to the little tree - shiver and quiver, my little tree, silver and gold throw down over me. And now the bird threw down to her a dress which was more splendid and magnificent than any she had yet had, and the slippers were golden. And when she went to the festival in the dress, no one knew how to speak for astonishment. The king's son danced with her only, and if any one invited her to dance, he said this is my partner. When evening came, cinderella wished to leave, and the king's son was anxious to go with her, but she escaped from him so quickly that he could not follow her. The king's son, however, had employed a ruse, and had caused the whole staircase to be smeared with pitch, and there, when she ran down, had the maiden's left slipper remained stuck. The king's son picked it up, and it was small and dainty, and all golden. Next morning, he went with it to the father, and said to him, no one shall be my wife but she whose foot this golden slipper fits. Then were the two sisters glad, for they had pretty feet. The eldest went with the shoe into her room and wanted to try it on, and her mother stood by. But she",
"afterwards the turtle-doves, and at length all the birds beneath the sky, came whirring and crowding in, and alighted amongst the ashes. And the doves nodded with their heads and began pick, pick, pick, pick, and the others began also pick, pick, pick, pick, and gathered all the good seeds into the dishes, and before half an hour was over they had already finished, and all flew out again. Then the maiden was delighted, and believed that she might now go with them to the wedding. But the step-mother said, all this will not help. You cannot go with us, for you have no clothes and can not dance. We should be ashamed of you. On this she turned her back on cinderella, and hurried away with her two proud daughters. As no one was now at home, cinderella went to her mother's grave beneath the hazel-tree, and cried - shiver and quiver, little tree, silver and gold throw down over me. Then the bird threw a gold and silver dress down to her, and slippers embroidered with silk and silver. She put on the dress with all speed, and went to the wedding. Her step-sisters and the step-mother however did not know her, and thought she must be a foreign princess, for she looked so beautiful in the golden dress. They never once thought of cinderella, and believed that she was sitting at home in the dirt, picking lentils out of the ashes. The prince approached her, took her by the hand and danced with her. He would dance with no other maiden, and never let loose of her hand, and if any one else came to invite her, he said, this is my partner. She danced till it was evening, and then she wanted to go home. But the king's son said, I will go with you and bear you company, for he wished to see to whom the beautiful maiden belonged.",
"The wife of a rich man fell sick, and as she felt that her end was drawing near, she called her only daughter to her bedside and said, dear child, be good and pious, and then the good God will always protect you, and I will look down on you from heaven and be near you. Thereupon she closed her eyes and departed. Every day the maiden went out to her mother's grave, and wept, and she remained pious and good. When winter came the snow spread a white sheet over the grave, and by the time the spring sun had drawn it off again, the man had taken another wife. The woman had brought with her into the house two daughters, who were beautiful and fair of face, but vile and black of heart. Now began a bad time for the poor step-child. Is the stupid goose to sit in the parlor with us, they said. He who wants to eat bread must earn it. Out with the kitchen-wench. They took her pretty clothes away from her, put an old grey bedgown on her, and gave her wooden shoes. Just look at the proud princess, how decked out she is, they cried, and laughed, and led her into the kitchen. There she had to do hard work from morning till night, get up before daybreak, carry water, light fires, cook and wash. Besides this, the sisters did her every imaginable injury - they mocked her and emptied her peas and lentils into the ashes, so that she was forced to sit and pick them out again. In the evening when she had worked till she was weary she had no bed to go to, but had to sleep by the hearth in the cinders. And as on that account she always looked dusty and dirty, they called her cinderella. It happened that the father was once going to the fair, and he asked his two step-daughters what he should bring back for them.",
"She escaped from him, however, and sprang into the pigeon-house. The king's son waited until her father came, and then he told him that the unknown maiden had leapt into the pigeon-house. The old man thought, can it be cinderella. And they had to bring him an axe and a pickaxe that he might hew the pigeon-house to pieces, but no one was inside it. And when they got home cinderella lay in her dirty clothes among the ashes, and a dim little oil-lamp was burning on the mantle-piece, for cinderella had jumped quickly down from the back of the pigeon-house and had run to the little hazel-tree, and there she had taken off her beautiful clothes and laid them on the grave, and the bird had taken them away again, and then she had seated herself in the kitchen amongst the ashes in her grey gown. Next day when the festival began afresh, and her parents and the step-sisters had gone once more, cinderella went to the hazel-tree and said - shiver and quiver, my little tree, silver and gold throw down over me. Then the bird threw down a much more beautiful dress than on the preceding day. And when cinderella appeared at the wedding in this dress, every one was astonished at her beauty. The king's son had waited until she came, and instantly took her by the hand and danced with no one but her. When others came and invited her, he said, this is my partner. When evening came she wished to leave, and the king's son followed her and wanted to see into which house she went. But she sprang away from him, and into the garden behind the house. Therein stood a beautiful tall tree on which hung the most magnificent pears. She clambered so nimbly between the branches like a squirrel that the king's son did not know where she was gone. He waited until her father came, and"]
:::
:::
