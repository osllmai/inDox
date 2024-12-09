[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/osllmai/inDoxRag/blob/master/Demo/hf_mistral_SimpleReader.ipynb)

# How to use IndoxRag Retrieval Augmentation for PDF files

In this notebook, we will demonstrate how to handle `inDoxRag` as system
for question answering system with open source models which are
available on internet like `Mistral`. so firstly you should buil
environment variables and API keys in Python using the `dotenv` library.

**Note**: Because we are using **HuggingFace** models you need to define
your `HUGGINGFACE_API_KEY` in `.env` file. This allows us to keep our
API keys and other sensitive information out of our codebase, enhancing
security and maintainability.
:::

::: {#4deb2abc71a048be .cell .code execution_count="8" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="4deb2abc71a048be" outputId="7d64d807-ed35-4ea1-fe15-fe5479ca796d"}
``` python
!pip install indoxRag
!pip install chromadb
!pip install semantic_text_splitter
!pip install sentence-transformers
```


::: {#2d6948cfe8ce7b88 .cell .code execution_count="2" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="2d6948cfe8ce7b88" outputId="e4e00368-40e9-46e3-b200-14b8bbc229f0"}
``` python
!wget https://raw.githubusercontent.com/osllmai/inDoxRag/master/Demo/sample.txt
```

::: {.output .stream .stdout}
    --2024-07-02 09:10:03--  https://raw.githubusercontent.com/osllmai/inDoxRag/master/Demo/sample.txt
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.109.133, 185.199.108.133, 185.199.110.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.109.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 14025 (14K) [text/plain]
    Saving to: ‘sample.txt’

    
sample.txt            0%[                    ]       0  --.-KB/s               
sample.txt          100%[===================>]  13.70K  --.-KB/s    in 0s      

    2024-07-02 09:10:03 (72.4 MB/s) - ‘sample.txt’ saved [14025/14025]
:::
:::

::: {#initial_id .cell .code execution_count="3" ExecuteTime="{\"end_time\":\"2024-07-02T07:44:38.098709Z\",\"start_time\":\"2024-07-02T07:44:38.086571Z\"}" collapsed="true" id="initial_id"}
``` python
import os
from dotenv import load_dotenv

load_dotenv()
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
```
:::

::: {#d97666fa4a136fa4 .cell .markdown id="d97666fa4a136fa4"}
### Import Essential Libraries

Then, we import essential libraries for our `IndoxRag` question answering
system:

-   `IndoxRetrievalAugmentation`: Enhances the retrieval process for
    better QA performance.
-   `MistralQA`: A powerful QA model from IndoxRag, built on top of the
    Hugging Face model.
-   `HuggingFaceEmbedding`: Utilizes Hugging Face embeddings for
    improved semantic understanding.
-   `SimpleLoadAndSplit`: A utility for loading and splitting PDF files.
:::

::: {#71cea6a5876fa5fe .cell .code execution_count="6" ExecuteTime="{\"end_time\":\"2024-07-02T07:44:47.832652Z\",\"start_time\":\"2024-07-02T07:44:38.872557Z\"}" id="71cea6a5876fa5fe"}
``` python
from indoxRag import IndoxRetrievalAugmentation
from indoxRag.llms import HuggingFaceModel
from indoxRag.embeddings import HuggingFaceEmbedding
from indoxRag.data_loader_splitter.SimpleLoadAndSplit import SimpleLoadAndSplit
```
:::

::: {#dfce2b023a435935 .cell .markdown id="dfce2b023a435935"}
### Building the IndoxRag System and Initializing Models

Next, we will build our `inDoxRag` system and initialize the Mistral
question answering model along with the embedding model. This setup will
allow us to leverage the advanced capabilities of IndoxRag for our question
answering tasks.
:::

::: {#be7f2eb137ea4f1b .cell .code execution_count="9" ExecuteTime="{\"end_time\":\"2024-07-02T07:44:51.558601Z\",\"start_time\":\"2024-07-02T07:44:47.833775Z\"}" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":369,\"referenced_widgets\":[\"f5a5036392c9462c903364eab1388cca\",\"2c9c482359c74bba8d35bd5a99b0b272\",\"fe107afe6cc74673afce75ad1fb61206\",\"4aa9bf7557d547f6a39fa93b96a8233d\",\"f3c97aebd0ce416f858a4abc8533bc1c\",\"a688174866694716a3f833a05761509f\",\"f2ef47b784a34c218cb5346d96183da6\",\"e7a1b752ee564ed18dafb4928bbad940\",\"a0ae506462fd4de9a831f3545cb4682e\",\"f33f1beca1f54fdd99a3d1ea7d03a0f1\",\"9b8f564b15da453ea811acde3f1b5eaa\",\"010ed74be3944660a12891dc607ca1a8\",\"6dbe57883b3a422abcc78cdc86faee0e\",\"adb60c0a86384134a8b0e9d1fb93dae6\",\"d759c9ae84284362ac1c197be51ef8fd\",\"823759359e8041a298e51ddeb9e718f8\",\"1f098f05988e4336b1e05c7bdf0530d7\",\"070514ed43e44b16ad0e5851deacafed\",\"b1237265df6b40f3837c4732cafee7fb\",\"5297f72091294256bd5f19263873827e\",\"3177fadb9d1a43bcba2c077f906b64ff\",\"2c724a13f5c041e693bf57d5740b43b2\",\"946464dc055e45f9bb9c1ead93247206\",\"44e038a01938429da9a44133590f5bfd\",\"8c83104d6ef24e5ca7a4838231ffece7\",\"b8047170243e429bbe35fcddff4697ce\",\"3136aac028444d09a16a3acdb05928b5\",\"527bc850f45b42a2a9ae6993d50ecd79\",\"7834cb453fd043f38b7e540bc70c90de\",\"86c85a928644439a9e6ded6a1ad86ebf\",\"ea89fbfff4e943eba0b769edabb9ffd8\",\"4b2f7cbd45e14c068d77aad2341b672c\",\"08cb878535274af8a7d771dbb71b58e7\",\"4112edbd6b044b569660dd38ffa3f27a\",\"0af588eacff84cbc997482f0878fa780\",\"8b4a84a48fc84cfb82cdfb244c5b4f05\",\"6cb894c218e5484ca128ff284f46aeec\",\"2302727f502e4cf5a8884d2f8d423183\",\"4c6896f5a2224f3090426ccfdf55f5ab\",\"e68009bb33394e6198f1109bacae6d67\",\"9ce868b2cc944e4b86eb97765acf7745\",\"c340eac0ba674bc9bf3736115cf0d7d4\",\"c1b7d9b59d184a37abb6e0c4f01f7469\",\"3d32d7acbdbc4de0ac73d0fbcd014acd\",\"c3dd1b712c324778a91a7c1750323ca2\",\"a7c53cf410054ad580f12d7f96fb7586\",\"e937971144b9448a80e76cc58224b174\",\"e2ff74c989f642d1b3f2d4e3e9043bc2\",\"a18a05fde8e146d5a2f8f66b9c5fe1d5\",\"ba9140e635c44689a717e6f146b40aa9\",\"072dac0b06a8452dbfcdfbc142beeab8\",\"98335a7da2b545db8098e2ace1f3ac1b\",\"3ce716123194429b9c444e2182d6093a\",\"46923b21aab24aba9253cd88e09656b8\",\"99ac55283f734f3cbc598e5b5ee228b7\",\"b7b98fd3242748a3941ab86879edb1d4\",\"a6306fd2f3564bdb88b25c4aee746bd3\",\"38c6199398824a4c880ef2e02374942c\",\"51198d2b64724ebda6b9ab58419b3cf9\",\"1d5ee00b1f334307b665043b88ec78cf\",\"33af708692764a40965ff40cc1679ae7\",\"9f3ce45cbe9f4db78bc5bee85ef3fbde\",\"fcd421736ea640d397175dab615d7131\",\"d077450e52b04d9e9a58e34c66459fcb\",\"ae47d975cc42451da6ad9f67051c0690\",\"e8bd35bd61c64d3f8f4646a9f51cba4b\",\"70ca761c554c49788f39ac022a1e0a0c\",\"c2d1474fc76040b7a783baac323b72eb\",\"260eda9b2c284046b3b13eec5fa3e681\",\"15a1950ddec644a0b4dbaf6fbd2d2f2c\",\"92592fda49944fd7a71e20cf988e0736\",\"01cce2aa951a4efab6b1de0e7c434841\",\"463a01c2f5d54f718bc7acb06322b410\",\"53fb6a95bd11498d99da873bcbc351ec\",\"b91a4d38baff4c7292eda4e41e306a8e\",\"762b71dde71740d3b266b30ee9cf7d0c\",\"77c48755a5584564b6d69ee6e59742f2\",\"3c540818f4fb4c34a3e1bb92d623e356\",\"b4d20af60de343c8bd7a559146f33288\",\"62a96756539a4c9c8b37a3880cf60856\",\"3ed60db66bfd4b7a9b6ea838bbf7cde0\",\"b229d81476d04f5a9a8b208ee50932e9\",\"9d7db8511d2043368f72994dcddca546\",\"585fc10a270144899f32c66f61884c25\",\"935e798098c74c0bb7b20a400c72a604\",\"a97b422055da4cf0b801e214a9c1d615\",\"115c6e6bc812404ead523830594419b0\",\"30ef7de6651e4fb19792027e9c5560f4\",\"2782e5da9ae14a2286373625261c9ddf\",\"cdd792528ab6407b9cb3504114d0b990\",\"1c64f40e0bfd4bcba077c0488f87d799\",\"985a161dcc784e748f757f7e7b0f4b72\",\"c309ce7dc72b4229bebe1c24932f2509\",\"671ef87d01514bf19c85827e6a963c03\",\"d125bf45f8054f0298ce128ac7c80f6f\",\"f28e3776dda74ce5a99a4f44bcad7fc4\",\"eb02f73ebda64b05aba04532c70fbfe2\",\"11b857b5dfb647eb90b9b3bb75089819\",\"fcb60e61fd8046f498aac8ff4bc6ed90\",\"8ec1f824a2d6489ba28811b1ebc233db\",\"c611f11954d94f2ea08bf3f5e251513b\",\"aef9e06b527b4c09a39c92613bda68e2\",\"4110c8d535dc4e5db696827274175581\",\"458a16a9cb864cfeb67d4fd8caa15b7c\",\"549c647ee0144ec1b99264eb7b6c5b5e\",\"e410b6dde03e4c9582b14fcf86351eab\",\"ea9de2ed992f45a58b3f9774d4a27fe5\",\"5c87e1522e8043bab149034a0849118c\",\"c2ab2050de7e4c6d95ce0209b7704853\",\"1677b83b6f7941288ffea7e26f1839a1\",\"5482f2e93f464564957f3260daed3bd2\",\"1d97c86d31b34be7ac7590d28858a81f\",\"ac4c0e5652de415f9887ff795bded3ac\",\"e3f7757502a144158e9b3bb8f0f8cd4c\",\"4fa1d4de5e794d38b2f1ef1362bc0056\",\"35a7be45599b46a0aecb73ee79b6bef0\",\"74f50b06501c4bcc8209b0541eaaad02\",\"f04994abe48b44c8ae1d40b2000032b5\",\"e6c0b3f713a2412db68aca62bee3124c\",\"0a01dec7cd7042e3806fe5db88fe68ae\",\"1b367c3e496948909341c2f15a9e146d\"]}" id="be7f2eb137ea4f1b" outputId="f774310a-e877-4983-b8f0-b3c97269f97f"}
``` python
indoxRag = IndoxRetrievalAugmentation()
mistral_qa = HuggingFaceModel(api_key=HUGGINGFACE_API_KEY,model="mistralai/Mistral-7B-Instruct-v0.2")
embed = HuggingFaceEmbedding(model="multi-qa-mpnet-base-cos-v1")
```

::: {.output .display_data}
``` json
{"model_id":"f5a5036392c9462c903364eab1388cca","version_major":2,"version_minor":0}
```
:::

::: {.output .display_data}
``` json
{"model_id":"010ed74be3944660a12891dc607ca1a8","version_major":2,"version_minor":0}
```
:::

::: {.output .display_data}
``` json
{"model_id":"946464dc055e45f9bb9c1ead93247206","version_major":2,"version_minor":0}
```
:::

::: {.output .display_data}
``` json
{"model_id":"4112edbd6b044b569660dd38ffa3f27a","version_major":2,"version_minor":0}
```
:::

::: {.output .display_data}
``` json
{"model_id":"c3dd1b712c324778a91a7c1750323ca2","version_major":2,"version_minor":0}
```
:::

::: {.output .display_data}
``` json
{"model_id":"b7b98fd3242748a3941ab86879edb1d4","version_major":2,"version_minor":0}
```
:::

::: {.output .display_data}
``` json
{"model_id":"70ca761c554c49788f39ac022a1e0a0c","version_major":2,"version_minor":0}
```
:::

::: {.output .display_data}
``` json
{"model_id":"3c540818f4fb4c34a3e1bb92d623e356","version_major":2,"version_minor":0}
```
:::

::: {.output .display_data}
``` json
{"model_id":"2782e5da9ae14a2286373625261c9ddf","version_major":2,"version_minor":0}
```
:::

::: {.output .display_data}
``` json
{"model_id":"8ec1f824a2d6489ba28811b1ebc233db","version_major":2,"version_minor":0}
```
:::

::: {.output .display_data}
``` json
{"model_id":"5482f2e93f464564957f3260daed3bd2","version_major":2,"version_minor":0}
```
:::
:::

::: {#70a4fe8c3d39b341 .cell .markdown id="70a4fe8c3d39b341"}
### Setting Up Reference Directory and File Path

To demonstrate the capabilities of our IndoxRag question answering system,
we will use a sample directory. This directory will contain our
reference data, which we will use for testing and evaluation.

First, we specify the path to our sample file. In this case, we are
using a file named `sample.txt` located in our working directory. This
file will serve as our reference data for the subsequent steps.

Let\'s define the file path for our reference data.
:::

::: {#de47eb24481ec6f0 .cell .code}
``` python
!wget https://raw.githubusercontent.com/osllmai/inDoxRag/master/Demo/sample.txt
```
:::

::: {#7d72d5ab31985758 .cell .code execution_count="10" ExecuteTime="{\"end_time\":\"2024-07-02T07:44:51.563008Z\",\"start_time\":\"2024-07-02T07:44:51.559605Z\"}" id="7d72d5ab31985758"}
``` python
file_path = "sample.txt"
```
:::

::: {#5c474890b4eb337 .cell .markdown id="5c474890b4eb337"}
### Chunking Reference Data with UnstructuredLoadAndSplit

To effectively utilize our reference data, we need to process and chunk
it into manageable parts. This ensures that our question answering
system can efficiently handle and retrieve relevant information.

We use the `SimpleLoadAndSplit` utility for this task. This tool allows
us to load the PDF files and split it into smaller chunks. This process
enhances the performance of our retrieval and QA models by making the
data more accessible and easier to process. We are using
\'bert-base-uncased\' model for splitting data.

In this step, we define the file path for our reference data and use
`SimpleLoadAndSplit` to chunk the data with a maximum chunk size of 200
characters. Also we can handle to remove stop words or not by
initializing `remove-sword` parameter.

Let\'s proceed with chunking our reference data.
:::

::: {#f45144aba717b77d .cell .code execution_count="11" ExecuteTime="{\"end_time\":\"2024-07-02T07:44:52.866553Z\",\"start_time\":\"2024-07-02T07:44:51.564014Z\"}" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":49,\"referenced_widgets\":[\"35a00d66d49f49b491287c686dfd6d9c\",\"0fb451900f6c49f3813a0708760d8db5\",\"6108a37bb56d4406947badb9e972bf35\",\"4198edcd7c654199b74deb5aa56ff2e6\",\"6b9eb0e710a646f28db2dcac8ad61820\",\"79f2fdb553be4edb9a9f3e82006f9355\",\"54caa58d86a84fb6b292a0dc7c1a6164\",\"237c618eeac047679c10714d37126997\",\"1ccdfdb963fb49d8bfe522545d198d7f\",\"889105b138bf413fb6658cf96e86ad38\",\"39503b49c47f4d268cd3fca8a41a82f3\"]}" id="f45144aba717b77d" outputId="5f16b724-f1bf-4f25-d10d-315c87dcc18b"}
``` python
simpleLoadAndSplit = SimpleLoadAndSplit(file_path="sample.txt",remove_sword=False,max_chunk_size=200)
docs = simpleLoadAndSplit.load_and_chunk()
```

::: {.output .display_data}
``` json
{"model_id":"35a00d66d49f49b491287c686dfd6d9c","version_major":2,"version_minor":0}
```
:::
:::

::: {#449dd09782582fa7 .cell .code execution_count="12" ExecuteTime="{\"end_time\":\"2024-07-02T07:44:52.872717Z\",\"start_time\":\"2024-07-02T07:44:52.867559Z\"}" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="449dd09782582fa7" outputId="e7b1e18b-9d3f-4464-c25f-c62e66db24d6"}
``` python
docs
```

::: {.output .execute_result execution_count="12"}
    ["The wife of a rich man fell sick, and as she felt that her end was drawing near, she called her only daughter to her bedside and said, dear child, be good and pious, and then the good God will always protect you, and I will look down on you from heaven and be near you.  Thereupon she closed her eyes and departed.  Every day the maiden went out to her mother's grave, and wept, and she remained pious and good.  When winter came the snow spread a white sheet over the grave, and by the time the spring sun had drawn it off again, the man had taken another wife. The woman had brought with her into the house two daughters, who were beautiful and fair of face, but vile and black of heart. Now began a bad time for the poor step-child.  Is the stupid goose to sit in the parlor with us, they said.  He who wants to eat bread",
     'must earn it.  Out with the kitchen-wench.  They took her pretty clothes away from her, put an old grey bedgown on her, and gave her wooden shoes.  Just look at the proud princess, how decked out she is, they cried, and laughed, and led her into the kitchen. There she had to do hard work from morning till night, get up before daybreak, carry water, light fires, cook and wash.  Besides this, the sisters did her every imaginable injury - they mocked her and emptied her peas and lentils into the ashes, so that she was forced to sit and pick them out again.  In the evening when she had worked till she was weary she had no bed to go to, but had to sleep by the hearth in the cinders.  And as on that account she always looked dusty and dirty, they called her cinderella. It happened that the father was once going to the fair, and he',
     "asked his two step-daughters what he should bring back for them. Beautiful dresses, said one, pearls and jewels, said the second. And you, cinderella, said he, what will you have.  Father break off for me the first branch which knocks against your hat on your way home.  So he bought beautiful dresses, pearls and jewels for his two step-daughters, and on his way home, as he was riding through a green thicket, a hazel twig brushed against him and knocked off his hat.  Then he broke off the branch and took it with him.  When he reached home he gave his step-daughters the things which they had wished for, and to cinderella he gave the branch from the hazel-bush.  Cinderella thanked him, went to her mother's grave and planted the branch on it, and wept so much that the tears fell down on it and watered it.  And it grew and became a handsome",
     "tree. Thrice a day cinderella went and sat beneath it, and wept and prayed, and a little white bird always came on the tree, and if cinderella expressed a wish, the bird threw down to her what she had wished for. It happened, however, that the king gave orders for a festival which was to last three days, and to which all the beautiful young girls in the country were invited, in order that his son might choose himself a bride.  When the two step-sisters heard that they too were to appear among the number, they were delighted, called cinderella and said, comb our hair for us, brush our shoes and fasten our buckles, for we are going to the wedding at the king's palace. Cinderella obeyed, but wept, because she too would have liked to go with them to the dance, and begged her step-mother to allow her to do so.  You go, cinderella, said she, covered in dust and",
     'dirt as you are, and would go to the festival.  You have no clothes and shoes, and yet would dance.  As, however, cinderella went on asking, the step-mother said at last, I have emptied a dish of lentils into the ashes for you, if you have picked them out again in two hours, you shall go with us.  The maiden went through the back-door into the garden, and called, you tame pigeons, you turtle-doves, and all you birds beneath the sky, come and help me to pick      the good into the pot,      the bad into the crop. Then two white pigeons came in by the kitchen window, and afterwards the turtle-doves, and at last all the birds beneath the sky, came whirring and crowding in, and alighted amongst the ashes. And the pigeons nodded with their heads and began pick, pick, pick, pick, and the rest began also pick, pick, pick, pick, and',
     'gathered all the good grains into the dish.  Hardly had one hour passed before they had finished, and all flew out again.  Then the girl took the dish to her step-mother, and was glad, and believed that now she would be allowed to go with them to the festival. But the step-mother said, no, cinderella, you have no clothes and you can not dance.  You would only be laughed at.  And as cinderella wept at this, the step-mother said, if you can pick two dishes of lentils out of the ashes for me in one hour, you shall go with us.  And she thought to herself, that she most certainly cannot do again.  When the step-mother had emptied the two dishes of lentils amongst the ashes, the maiden went through the back-door into the garden and cried, you tame pigeons, you turtle-doves, and all you birds beneath the sky, come and help me to pick      the good into the pot,',
     "the bad into the crop. Then two white pigeons came in by the kitchen-window, and afterwards the turtle-doves, and at length all the birds beneath the sky, came whirring and crowding in, and alighted amongst the ashes.  And the doves nodded with their heads and began pick, pick, pick, pick, and the others began also pick, pick, pick, pick, and gathered all the good seeds into the dishes, and before half an hour was over they had already finished, and all flew out again. Then the maiden was delighted, and believed that she might now go with them to the wedding.  But the step-mother said, all this will not help.  You cannot go with us, for you have no clothes and can not dance.  We should be ashamed of you.  On this she turned her back on cinderella, and hurried away with her two proud daughters. As no one was now at home, cinderella went to her mother's",
     'grave beneath the hazel-tree, and cried -      shiver and quiver, little tree,      silver and gold throw down over me. Then the bird threw a gold and silver dress down to her, and slippers embroidered with silk and silver.  She put on the dress with all speed, and went to the wedding.  Her step-sisters and the step-mother however did not know her, and thought she must be a foreign princess, for she looked so beautiful in the golden dress. They never once thought of cinderella, and believed that she was sitting at home in the dirt, picking lentils out of the ashes.  The prince approached her, took her by the hand and danced with her. He would dance with no other maiden, and never let loose of her hand, and if any one else came to invite her, he said, this is my partner. She danced till it was evening, and then she wanted to go home.',
     "But the king's son said, I will go with you and bear you company, for he wished to see to whom the beautiful maiden belonged. She escaped from him, however, and sprang into the pigeon-house.  The king's son waited until her father came, and then he told him that the unknown maiden had leapt into the pigeon-house.  The old man thought, can it be cinderella.  And they had to bring him an axe and a pickaxe that he might hew the pigeon-house to pieces, but no one was inside it.  And when they got home cinderella lay in her dirty clothes among the ashes, and a dim little oil-lamp was burning on the mantle-piece, for cinderella had jumped quickly down from the back of the pigeon-house and had run to the little hazel-tree, and there she had taken off her beautiful clothes and laid them on the grave, and the bird had",
     "taken them away again, and then she had seated herself in the kitchen amongst the ashes in her grey gown. Next day when the festival began afresh, and her parents and the step-sisters had gone once more, cinderella went to the hazel-tree and said -      shiver and quiver, my little tree,      silver and gold throw down over me. Then the bird threw down a much more beautiful dress than on the preceding day. And when cinderella appeared at the wedding in this dress, every one was astonished at her beauty.  The king's son had waited until she came, and instantly took her by the hand and danced with no one but her.  When others came and invited her, he said, this is my partner.  When evening came she wished to leave, and the king's son followed her and wanted to see into which house she went.  But she sprang away from him, and into the garden behind the house.  Therein stood a beautiful tall tree on",
     "which hung the most magnificent pears.  She clambered so nimbly between the branches like a squirrel that the king's son did not know where she was gone.  He waited until her father came, and said to him, the unknown maiden has escaped from me, and I believe she has climbed up the pear-tree.  The father thought, can it be cinderella.  And had an axe brought and cut the tree down, but no one was on it.  And when they got into the kitchen, cinderella lay there among the ashes, as usual, for she had jumped down on the other side of the tree, had taken the beautiful dress to the bird on the little hazel-tree, and put on her grey gown. On the third day, when the parents and sisters had gone away, cinderella went once more to her mother's grave and said to the little tree -      shiver and quiver, my little tree,      silver and gold throw down over me.",
     "And now the bird threw down to her a dress which was more splendid and magnificent than any she had yet had, and the slippers were golden.  And when she went to the festival in the dress, no one knew how to speak for astonishment.  The king's son danced with her only, and if any one invited her to dance, he said this is my partner. When evening came, cinderella wished to leave, and the king's son was anxious to go with her, but she escaped from him so quickly that he could not follow her.  The king's son, however, had employed a ruse, and had caused the whole staircase to be smeared with pitch, and there, when she ran down, had the maiden's left slipper remained stuck.  The king's son picked it up, and it was small and dainty, and all golden.  Next morning, he went with it to",
     "the father, and said to him, no one shall be my wife but she whose foot this golden slipper fits.  Then were the two sisters glad, for they had pretty feet.  The eldest went with the shoe into her room and wanted to try it on, and her mother stood by.  But she could not get her big toe into it, and the shoe was too small for her.  Then her mother gave her a knife and said, cut the toe off, when you are queen you will have no more need to go on foot.  The maiden cut the toe off, forced the foot into the shoe, swallowed the pain, and went out to the king's son.  Then he took her on his his horse as his bride and rode away with her.  They were obliged, however, to pass the grave, and there, on the hazel-tree, sat the two pigeons and cried -      turn and peep, turn and peep,      there's blood within the shoe,",
     "the shoe it is too small for her,      the true bride waits for you. Then he looked at her foot and saw how the blood was trickling from it.  He turned his horse round and took the false bride home again, and said she was not the true one, and that the other sister was to put the shoe on.  Then this one went into her chamber and got her toes safely into the shoe, but her heel was too large.  So her mother gave her a knife and said,  cut a bit off your heel, when you are queen you will have no more need to go on foot.  The maiden cut a bit off her heel, forced her foot into the shoe, swallowed the pain, and went out to the king's son.  He took her on his horse as his bride, and rode away with her, but when they passed by the hazel-tree, the two pigeons sat on it and cried -      turn and peep, turn and peep,",
     "there's blood within the shoe,      the shoe it is too small for her,      the true bride waits for you. He looked down at her foot and saw how the blood was running out of her shoe, and how it had stained her white stocking quite red.  Then he turned his horse and took the false bride home again.  This also is not the right one, said he, have you no other daughter.  No, said the man, there is still a little stunted kitchen-wench which my late wife left behind her, but she cannot possibly be the bride.  The king's son said he was to send her up to him, but the mother answered, oh, no, she is much too dirty, she cannot show herself.  But he absolutely insisted on it, and cinderella had to be called.  She first washed her hands and face clean, and then went and bowed down before the king's son, who gave her the golden shoe.  Then she",
     "seated herself on a stool, drew her foot out of the heavy wooden shoe, and put it into the slipper, which fitted like a glove.  And when she rose up and the king's son looked at her face he recognized the beautiful maiden who had danced with him and cried, that is the true bride.  The step-mother and the two sisters were horrified and became pale with rage, he, however, took cinderella on his horse and rode away with her.  As they passed by the hazel-tree, the two white doves cried -      turn and peep, turn and peep,      no blood is in the shoe,      the shoe is not too small for her,      the true bride rides with you, and when they had cried that, the two came flying down and placed themselves on cinderella's shoulders, one on the right, the other on the left, and remained sitting there. When the wedding with the king's son was to be celebrated, the",
     'two false sisters came and wanted to get into favor with cinderella and share her good fortune.  When the betrothed couple went to church, the elder was at the right side and the younger at the left, and the pigeons pecked out one eye from each of them.  Afterwards as they came back the elder was at the left, and the younger at the right, and then the pigeons pecked out the other eye from each.  And thus, for their wickedness and falsehood, they were punished with blindness all their days.']
:::
:::

::: {#28ba43c70a109fe4 .cell .markdown id="28ba43c70a109fe4"}
### Connecting Embedding Model to IndoxRag

With our reference data chunked and ready, the next step is to connect
our embedding model to the IndoxRag system. This connection enables the
system to leverage the embeddings for better semantic understanding and
retrieval performance.

We use the `connect_to_vectorstore` method to link the
`HuggingFaceEmbedding` model with our IndoxRag system. By specifying the
embeddings and a collection name, we ensure that our reference data is
appropriately indexed and stored, facilitating efficient retrieval
during the question-answering process.

Let\'s connect the embedding model to IndoxRag.
:::

::: {#fb0c5eb8341d52d3 .cell .code execution_count="13" ExecuteTime="{\"end_time\":\"2024-07-02T07:44:53.236317Z\",\"start_time\":\"2024-07-02T07:44:52.873727Z\"}" id="fb0c5eb8341d52d3"}
``` python
from indoxRag.vector_stores import ChromaVectorStore
db = ChromaVectorStore(collection_name="sample",embedding=embed)
```
:::

::: {#bbd77c136e780844 .cell .code execution_count="14" ExecuteTime="{\"end_time\":\"2024-07-02T07:44:53.242353Z\",\"start_time\":\"2024-07-02T07:44:53.237325Z\"}" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="bbd77c136e780844" outputId="3dc71f1c-96a0-4c0f-cb44-611f2db058c0"}
``` python
indoxRag.connect_to_vectorstore(vectorstore_database=db)
```

::: {.output .execute_result execution_count="14"}
    <indoxRag.vector_stores.Chroma.ChromaVectorStore at 0x7a1e30ccee90>
:::
:::

::: {#7a2de351e4cb5e24 .cell .markdown id="7a2de351e4cb5e24"}
### Storing Data in the Vector Store

After connecting our embedding model to the IndoxRag system, the next step
is to store our chunked reference data in the vector store. This process
ensures that our data is indexed and readily available for retrieval
during the question-answering process.

We use the `store_in_vectorstore` method to store the processed data in
the vector store. By doing this, we enhance the system\'s ability to
quickly access and retrieve relevant information based on the embeddings
generated earlier.

Let\'s proceed with storing the data in the vector store.
:::

::: {#8b53fee18caed89c .cell .code execution_count="15" ExecuteTime="{\"end_time\":\"2024-07-02T07:45:07.044107Z\",\"start_time\":\"2024-07-02T07:45:03.390792Z\"}" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="8b53fee18caed89c" outputId="ab39b1ac-03de-44e1-977b-b2157538f9a9"}
``` python
indoxRag.store_in_vectorstore(docs)
```

::: {.output .execute_result execution_count="15"}
    <indoxRag.vector_stores.Chroma.ChromaVectorStore at 0x7a1e30ccee90>
:::
:::

::: {#a2ac994fc0fb7ca0 .cell .markdown id="a2ac994fc0fb7ca0"}
## Query from RAG System with IndoxRag

With our Retrieval-Augmented Generation (RAG) system built using IndoxRag,
we are now ready to test it with a sample question. This test will
demonstrate how effectively our system can retrieve and generate
accurate answers based on the reference data stored in the vector store.

We\'ll use a sample query to test our system:

-   **Query**: \"How did Cinderella reach her happy ending?\"

This question will be processed by our IndoxRag system to retrieve relevant
information and generate an appropriate response.

Let\'s test our RAG system with the sample question
:::

::: {#6b9fcd8f902257ad .cell .code execution_count="16" ExecuteTime="{\"end_time\":\"2024-07-02T07:45:08.545059Z\",\"start_time\":\"2024-07-02T07:45:08.541085Z\"}" id="6b9fcd8f902257ad"}
``` python
query = "How cinderella reach her happy ending?"
```
:::

::: {#905f5aeb288a9ea3 .cell .markdown id="905f5aeb288a9ea3"}
Now that our Retrieval-Augmented Generation (RAG) system with IndoxRag is
fully set up, we can test it with a sample question. We\'ll use the
`invoke` submethod to get a response from the system.

The `invoke` method processes the query using the connected QA model and
retrieves relevant information from the vector store. It returns a list
where:

-   The first index contains the answer.
-   The second index contains the contexts and their respective scores.

We\'ll pass this query to the `invoke` method and print the response.
:::

::: {#5a87158e90e6cb32 .cell .code execution_count="17" ExecuteTime="{\"end_time\":\"2024-07-02T07:45:09.503644Z\",\"start_time\":\"2024-07-02T07:45:09.500273Z\"}" id="5a87158e90e6cb32"}
``` python
retriever = indoxRag.QuestionAnswer(vector_database=db,llm=mistral_qa,top_k=5)
```
:::

::: {#6fe8b6eff074cee .cell .code execution_count="18" ExecuteTime="{\"end_time\":\"2024-07-02T07:45:16.862965Z\",\"start_time\":\"2024-07-02T07:45:10.367688Z\"}" id="6fe8b6eff074cee"}
``` python
answer = retriever.invoke(query=query)
```
:::

::: {#8a5500901a37dcf4 .cell .code execution_count="19" ExecuteTime="{\"end_time\":\"2024-07-02T07:45:18.009646Z\",\"start_time\":\"2024-07-02T07:45:18.005532Z\"}" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":70}" id="8a5500901a37dcf4" outputId="db8b708c-4512-4350-d279-11d831804c8d"}
``` python
answer
```

::: {.output .execute_result execution_count="19"}
``` json
{"type":"string"}
```
:::
:::

::: {#4a0b652a31bba343 .cell .code execution_count="20" ExecuteTime="{\"end_time\":\"2024-07-02T07:45:30.087337Z\",\"start_time\":\"2024-07-02T07:45:30.081831Z\"}" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="4a0b652a31bba343" outputId="8e7742d1-19f2-449a-e455-15259515e760"}
``` python
context = retriever.context
context
```

::: {.output .execute_result execution_count="20"}
    ['must earn it.  Out with the kitchen-wench.  They took her pretty clothes away from her, put an old grey bedgown on her, and gave her wooden shoes.  Just look at the proud princess, how decked out she is, they cried, and laughed, and led her into the kitchen. There she had to do hard work from morning till night, get up before daybreak, carry water, light fires, cook and wash.  Besides this, the sisters did her every imaginable injury - they mocked her and emptied her peas and lentils into the ashes, so that she was forced to sit and pick them out again.  In the evening when she had worked till she was weary she had no bed to go to, but had to sleep by the hearth in the cinders.  And as on that account she always looked dusty and dirty, they called her cinderella. It happened that the father was once going to the fair, and he',
     "And now the bird threw down to her a dress which was more splendid and magnificent than any she had yet had, and the slippers were golden.  And when she went to the festival in the dress, no one knew how to speak for astonishment.  The king's son danced with her only, and if any one invited her to dance, he said this is my partner. When evening came, cinderella wished to leave, and the king's son was anxious to go with her, but she escaped from him so quickly that he could not follow her.  The king's son, however, had employed a ruse, and had caused the whole staircase to be smeared with pitch, and there, when she ran down, had the maiden's left slipper remained stuck.  The king's son picked it up, and it was small and dainty, and all golden.  Next morning, he went with it to",
     "there's blood within the shoe,      the shoe it is too small for her,      the true bride waits for you. He looked down at her foot and saw how the blood was running out of her shoe, and how it had stained her white stocking quite red.  Then he turned his horse and took the false bride home again.  This also is not the right one, said he, have you no other daughter.  No, said the man, there is still a little stunted kitchen-wench which my late wife left behind her, but she cannot possibly be the bride.  The king's son said he was to send her up to him, but the mother answered, oh, no, she is much too dirty, she cannot show herself.  But he absolutely insisted on it, and cinderella had to be called.  She first washed her hands and face clean, and then went and bowed down before the king's son, who gave her the golden shoe.  Then she",
     "tree. Thrice a day cinderella went and sat beneath it, and wept and prayed, and a little white bird always came on the tree, and if cinderella expressed a wish, the bird threw down to her what she had wished for. It happened, however, that the king gave orders for a festival which was to last three days, and to which all the beautiful young girls in the country were invited, in order that his son might choose himself a bride.  When the two step-sisters heard that they too were to appear among the number, they were delighted, called cinderella and said, comb our hair for us, brush our shoes and fasten our buckles, for we are going to the wedding at the king's palace. Cinderella obeyed, but wept, because she too would have liked to go with them to the dance, and begged her step-mother to allow her to do so.  You go, cinderella, said she, covered in dust and",
     'grave beneath the hazel-tree, and cried -      shiver and quiver, little tree,      silver and gold throw down over me. Then the bird threw a gold and silver dress down to her, and slippers embroidered with silk and silver.  She put on the dress with all speed, and went to the wedding.  Her step-sisters and the step-mother however did not know her, and thought she must be a foreign princess, for she looked so beautiful in the golden dress. They never once thought of cinderella, and believed that she was sitting at home in the dirt, picking lentils out of the ashes.  The prince approached her, took her by the hand and danced with her. He would dance with no other maiden, and never let loose of her hand, and if any one else came to invite her, he said, this is my partner. She danced till it was evening, and then she wanted to go home.']
:::
:::

::: {#ef42aa27b31243a .cell .code ExecuteTime="{\"end_time\":\"2024-07-02T07:45:34.018349Z\",\"start_time\":\"2024-07-02T07:45:34.014766Z\"}" id="ef42aa27b31243a"}
``` python
```
:::

::: {#f32928ee1157509f .cell .markdown id="f32928ee1157509f"}
## Evaluation
:::

::: {#d6362d3145cb9f05 .cell .code ExecuteTime="{\"end_time\":\"2024-07-02T07:46:07.503919Z\",\"start_time\":\"2024-07-02T07:46:05.038476Z\"}" id="d6362d3145cb9f05"}
``` python
from indoxRag.evaluation import Evaluation
evaluator = Evaluation(["BertScore"])
```
:::

::: {#717b701e7958a0c .cell .code ExecuteTime="{\"end_time\":\"2024-07-02T07:46:14.343498Z\",\"start_time\":\"2024-07-02T07:46:13.455591Z\"}" id="717b701e7958a0c"}
``` python
inputs = {
    "question" : query,
    "answer" : answer,
    "context" : context
}
result = evaluator(inputs)
```
:::

::: {#73f30dd1c899a55d .cell .code ExecuteTime="{\"end_time\":\"2024-07-02T07:46:19.019465Z\",\"start_time\":\"2024-07-02T07:46:19.011378Z\"}" id="73f30dd1c899a55d" outputId="f1f3f407-5ada-45f8-9a30-60a1b3542554"}
``` python
result
```

::: {.output .execute_result execution_count="18"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Precision</th>
      <td>0.547425</td>
    </tr>
    <tr>
      <th>Recall</th>
      <td>0.482575</td>
    </tr>
    <tr>
      <th>F1-score</th>
      <td>0.507377</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#6e2160b7c2e28cb1 .cell .code id="6e2160b7c2e28cb1"}
``` python
```
:::
