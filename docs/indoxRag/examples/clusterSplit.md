---
title: Load And Split With Clustering
---

```
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
```

## Initial Setup

The following imports are essential for setting up the Indox application. These imports include the main Indox retrieval augmentation module, question-answering models, embeddings, and data loader splitter.



```
from indox import IndoxRetrievalAugmentation
from indox.llms import OpenAiQA
from indox.embeddings import OpenAiEmbedding
from indox.data_loader_splitter import ClusteredSplit
```

In this step, we initialize the Indox Retrieval Augmentation, the QA model, and the embedding model. Note that the models used for QA and embedding can vary depending on the specific requirements.



```
Indox = IndoxRetrievalAugmentation()
qa_model = OpenAiQA(api_key=OPENAI_API_KEY,model="gpt-3.5-turbo-0125")
embed = OpenAiEmbedding(api_key=OPENAI_API_KEY,model="text-embedding-3-small")
```

## Data Loader Setup

We set up the data loader using the `ClusteredSplit` class. This step involves loading documents, configuring embeddings, and setting options for processing the text.



```
loader_splitter = ClusteredSplit(file_path="sample.txt",embeddings=embed,remove_sword=False,re_chunk=False,chunk_size=300,use_openai_summary=True)
```


```
docs = loader_splitter.load_and_chunk()
```

    --Generated 1 clusters--
    

## Vector Store Connection and Document Storage

In this step, we connect the Indox application to the vector store and store the processed documents.



```
from indox.vector_stores import ChromaVectorStore
db = ChromaVectorStore(collection_name="sample",embedding=embed)
```


```
Indox.connect_to_vectorstore(db)
```




    <indox.vector_stores.Chroma.ChromaVectorStore at 0x215d56a6c00>




```
Indox.store_in_vectorstore(docs)
```




    <indox.vector_stores.Chroma.ChromaVectorStore at 0x215d56a6c00>



## Querying and Interpreting the Response

In this step, we query the Indox application with a specific question and use the QA model to get the response. The response is a tuple where the first element is the answer and the second element contains the retrieved context with their cosine scores.
response[0] contains the answer
response[1] contains the retrieved context with their cosine scores



```
retriever = Indox.QuestionAnswer(vector_database=db,llm=qa_model,top_k=5)
```


```
retriever.invoke(query="How cinderella reach happy ending?")
```




    "Cinderella reached her happy ending through her kindness, perseverance, and the magical assistance she received. Despite being mistreated by her stepmother and stepsisters, Cinderella remained kind and pure of heart. With the help of a little bird and her mother's grave, she was able to attend the royal ball where the prince fell in love with her. Even though she had to escape from the prince, he searched for her and eventually found her with the help of the golden slipper she left behind. The prince then declared that he would only marry the woman whose foot fit the golden slipper, leading to Cinderella's ultimate happy ending as she was the only one whose foot fit the slipper."




```
retriever.context
```




    ["They never once thought of cinderella, and believed that she was sitting at home in the dirt, picking lentils out of the ashes   The prince approached her, took her by the hand and danced with her He would dance with no other maiden, and never let loose of her hand, and if any one else came to invite her, he said, this is my partner She danced till it was evening, and then she wanted to go home But the king's son said, I will go with you and bear you company, for he wished to see to whom the beautiful maiden belonged She escaped from him, however, and sprang into the pigeon-house   The king's son waited until her father came, and then he told him that the unknown maiden had leapt into the pigeon-house   The old man thought, can it be cinderella   And they had to bring him an axe and a pickaxe that he might hew the pigeon-house to pieces, but no one was inside it   And when they got home cinderella lay in her dirty clothes among the ashes, and a dim little oil-lamp was burning on the mantle-piece, for cinderella had jumped quickly down from the back of the pigeon-house and had run to the little hazel-tree, and there she had taken off her beautiful clothes and laid them on the grave, and the bird had taken them away again, and then she had seated herself in the kitchen amongst the ashes in her grey gown",
     "The documentation provided describes the classic fairy tale of Cinderella. It tells the story of a young girl, Cinderella, who is mistreated by her stepmother and stepsisters after her mother's death. Despite their cruelty, Cinderella remains kind and pure of heart. Through magical assistance from a little bird and her mother's grave, Cinderella is able to attend the royal ball where the prince falls in love with her. When she escapes, the prince searches for her and finally finds her with the",
     "had jumped down on the other side of the tree, had taken the beautiful dress to the bird on the little hazel-tree, and put on her grey gown On the third day, when the parents and sisters had gone away, cinderella went once more to her mother's grave and said to the little tree -      shiver and quiver, my little tree,      silver and gold throw down over me And now the bird threw down to her a dress which was more splendid and magnificent than any she had yet had, and the slippers were golden   And when she went to the festival in the dress, no one knew how to speak for astonishment   The king's son danced with her only, and if any one invited her to dance, he said this is my partner When evening came, cinderella wished to leave, and the king's son was anxious to go with her, but she escaped from him so quickly that he could not follow her   The king's son, however, had employed a ruse, and had caused the whole staircase to be smeared with pitch, and there, when she ran down, had the maiden's left slipper remained stuck   The king's son picked it up, and it was small and dainty, and all golden   Next morning, he went with it to the father, and said to him, no one shall be my wife but she whose foot this golden slipper fits   Then were the two sisters glad,",
     "and emptied her peas and lentils into the ashes, so that she was forced to sit and pick them out again   In the evening when she had worked till she was weary she had no bed to go to, but had to sleep by the hearth in the cinders   And as on that account she always looked dusty and dirty, they called her cinderella It happened that the father was once going to the fair, and he asked his two step-daughters what he should bring back for them Beautiful dresses, said one, pearls and jewels, said the second And you, cinderella, said he, what will you have   Father break off for me the first branch which knocks against your hat on your way home   So he bought beautiful dresses, pearls and jewels for his two step-daughters, and on his way home, as he was riding through a green thicket, a hazel twig brushed against him and knocked off his hat   Then he broke off the branch and took it with him   When he reached home he gave his step-daughters the things which they had wished for, and to cinderella he gave the branch from the hazel-bush   Cinderella thanked him, went to her mother's grave and planted the branch on it, and wept so much that the tears fell down on it and watered it   And it grew and became a handsome tree  Thrice a day cinderella went and sat beneath it, and wept and",
     "prayed, and a little white bird always came on the tree, and if cinderella expressed a wish, the bird threw down to her what she had wished for It happened, however, that the king gave orders for a festival which was to last three days, and to which all the beautiful young girls in the country were invited, in order that his son might choose himself a bride   When the two step-sisters heard that they too were to appear among the number, they were delighted, called cinderella and said, comb our hair for us, brush our shoes and fasten our buckles, for we are going to the wedding at the king's palace Cinderella obeyed, but wept, because she too would have liked to go with them to the dance, and begged her step-mother to allow her to do so   You go, cinderella, said she, covered in dust and dirt as you are, and would go to the festival   You have no clothes and shoes, and yet would dance   As, however, cinderella went on asking, the step-mother said at last, I have emptied a dish of lentils into the ashes for you, if you have picked them out again in two hours, you shall go with us   The maiden went through the back-door into the garden, and called, you tame pigeons, you turtle-doves, and all you birds beneath the sky, come and help me to pick"]




```

```
