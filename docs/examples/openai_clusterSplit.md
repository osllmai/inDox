---
jupyter:
  colab:
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
  nbformat: 4
  nbformat_minor: 5
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/osllmai/inDox/blob/master/Demo/openai_clusterSplit.ipynb)

# Load And Split With Clustering


::: {#lj_IWpNvkRbD .cell .code execution_count="1" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="lj_IWpNvkRbD" outputId="01134cda-fefa-4c13-847c-d3592dd2547d"}
``` python
!pip install indox
!pip install openai
!pip install chromadb
```

::: {.output .stream .stdout}
    Collecting indox
      Downloading Indox-0.1.11-py3-none-any.whl (74 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 74.8/74.8 kB 1.1 MB/s eta 0:00:00
     indox)
      Downloading langchain-0.2.6-py3-none-any.whl (975 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 975.5/975.5 kB 15.8 MB/s eta 0:00:00
    munity (from indox)
      Downloading langchain_community-0.2.6-py3-none-any.whl (2.2 MB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.2/2.2 MB 37.5 MB/s eta 0:00:00
     indox)
      Downloading langchain_core-0.2.10-py3-none-any.whl (332 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 332.8/332.8 kB 20.0 MB/s eta 0:00:00
    istralai (from indox)
      Downloading langchain_mistralai-0.1.9-py3-none-any.whl (13 kB)
    Collecting langchain-openai (from indox)
      Downloading langchain_openai-0.1.13-py3-none-any.whl (45 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 45.9/45.9 kB 3.0 MB/s eta 0:00:00
     indox)
      Downloading langgraph-0.1.4-py3-none-any.whl (88 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 88.9/88.9 kB 7.5 MB/s eta 0:00:00
    arkdown (from indox)
      Downloading latex2markdown-0.2.1.tar.gz (161 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 161.7/161.7 kB 13.3 MB/s eta 0:00:00
    etadata (setup.py) ... ent already satisfied: nltk in /usr/local/lib/python3.10/dist-packages (from indox) (3.8.1)
    Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from indox) (1.25.2)
    Requirement already satisfied: pandas~=2.0.3 in /usr/local/lib/python3.10/dist-packages (from indox) (2.0.3)
    Collecting PyPDF2 (from indox)
      Downloading pypdf2-3.0.1-py3-none-any.whl (232 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 232.6/232.6 kB 17.4 MB/s eta 0:00:00
    ent already satisfied: Requests in /usr/local/lib/python3.10/dist-packages (from indox) (2.31.0)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from indox) (67.7.2)
    Requirement already satisfied: tenacity in /usr/local/lib/python3.10/dist-packages (from indox) (8.4.1)
    Collecting tiktoken (from indox)
      Downloading tiktoken-0.7.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.1 MB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.1/1.1 MB 21.3 MB/s eta 0:00:00
    ent already satisfied: tokenizers in /usr/local/lib/python3.10/dist-packages (from indox) (0.19.1)
    Collecting umap-learn (from indox)
      Downloading umap_learn-0.5.6-py3-none-any.whl (85 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 85.7/85.7 kB 7.8 MB/s eta 0:00:00
     indox)
      Downloading unstructured-0.14.9-py3-none-any.whl (2.1 MB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 39.9 MB/s eta 0:00:00
     indox)
      Downloading utils-1.0.2.tar.gz (13 kB)
      Preparing metadata (setup.py) ...  indox)
      Downloading python_dotenv-1.0.1-py3-none-any.whl (19 kB)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas~=2.0.3->indox) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas~=2.0.3->indox) (2023.4)
    Requirement already satisfied: tzdata>=2022.1 in /usr/local/lib/python3.10/dist-packages (from pandas~=2.0.3->indox) (2024.1)
    Requirement already satisfied: PyYAML>=5.3 in /usr/local/lib/python3.10/dist-packages (from langchain->indox) (6.0.1)
    Requirement already satisfied: SQLAlchemy<3,>=1.4 in /usr/local/lib/python3.10/dist-packages (from langchain->indox) (2.0.31)
    Requirement already satisfied: aiohttp<4.0.0,>=3.8.3 in /usr/local/lib/python3.10/dist-packages (from langchain->indox) (3.9.5)
    Requirement already satisfied: async-timeout<5.0.0,>=4.0.0 in /usr/local/lib/python3.10/dist-packages (from langchain->indox) (4.0.3)
    Collecting langchain-text-splitters<0.3.0,>=0.2.0 (from langchain->indox)
      Downloading langchain_text_splitters-0.2.2-py3-none-any.whl (25 kB)
    Collecting langsmith<0.2.0,>=0.1.17 (from langchain->indox)
      Downloading langsmith-0.1.82-py3-none-any.whl (127 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 127.4/127.4 kB 6.8 MB/s eta 0:00:00
    ent already satisfied: pydantic<3,>=1 in /usr/local/lib/python3.10/dist-packages (from langchain->indox) (2.7.4)
    Collecting jsonpatch<2.0,>=1.33 (from langchain-core->indox)
      Downloading jsonpatch-1.33-py2.py3-none-any.whl (12 kB)
    Requirement already satisfied: packaging<25,>=23.2 in /usr/local/lib/python3.10/dist-packages (from langchain-core->indox) (24.1)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from Requests->indox) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from Requests->indox) (3.7)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from Requests->indox) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from Requests->indox) (2024.6.2)
    Collecting dataclasses-json<0.7,>=0.5.7 (from langchain-community->indox)
      Downloading dataclasses_json-0.6.7-py3-none-any.whl (28 kB)
    Collecting httpx<1,>=0.25.2 (from langchain-mistralai->indox)
      Downloading httpx-0.27.0-py3-none-any.whl (75 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 75.6/75.6 kB 7.3 MB/s eta 0:00:00
     langchain-mistralai->indox)
      Downloading httpx_sse-0.4.0-py3-none-any.whl (7.8 kB)
    Requirement already satisfied: huggingface-hub<1.0,>=0.16.4 in /usr/local/lib/python3.10/dist-packages (from tokenizers->indox) (0.23.4)
    Collecting openai<2.0.0,>=1.32.0 (from langchain-openai->indox)
      Downloading openai-1.35.7-py3-none-any.whl (327 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 327.5/327.5 kB 27.2 MB/s eta 0:00:00
    ent already satisfied: regex>=2022.1.18 in /usr/local/lib/python3.10/dist-packages (from tiktoken->indox) (2024.5.15)
    Requirement already satisfied: click in /usr/local/lib/python3.10/dist-packages (from nltk->indox) (8.1.7)
    Requirement already satisfied: joblib in /usr/local/lib/python3.10/dist-packages (from nltk->indox) (1.4.2)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from nltk->indox) (4.66.4)
    Requirement already satisfied: scipy>=1.3.1 in /usr/local/lib/python3.10/dist-packages (from umap-learn->indox) (1.11.4)
    Requirement already satisfied: scikit-learn>=0.22 in /usr/local/lib/python3.10/dist-packages (from umap-learn->indox) (1.2.2)
    Requirement already satisfied: numba>=0.51.2 in /usr/local/lib/python3.10/dist-packages (from umap-learn->indox) (0.58.1)
    Collecting pynndescent>=0.5 (from umap-learn->indox)
      Downloading pynndescent-0.5.13-py3-none-any.whl (56 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 56.9/56.9 kB 2.5 MB/s eta 0:00:00
    ent already satisfied: chardet in /usr/local/lib/python3.10/dist-packages (from unstructured->indox) (5.2.0)
    Collecting filetype (from unstructured->indox)
      Downloading filetype-1.2.0-py2.py3-none-any.whl (19 kB)
    Collecting python-magic (from unstructured->indox)
      Downloading python_magic-0.4.27-py2.py3-none-any.whl (13 kB)
    Requirement already satisfied: lxml in /usr/local/lib/python3.10/dist-packages (from unstructured->indox) (4.9.4)
    Requirement already satisfied: tabulate in /usr/local/lib/python3.10/dist-packages (from unstructured->indox) (0.9.0)
    Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.10/dist-packages (from unstructured->indox) (4.12.3)
    Collecting emoji (from unstructured->indox)
      Downloading emoji-2.12.1-py3-none-any.whl (431 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 431.4/431.4 kB 31.8 MB/s eta 0:00:00
     unstructured->indox)
      Downloading python_iso639-2024.4.27-py3-none-any.whl (274 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 274.7/274.7 kB 24.3 MB/s eta 0:00:00
     unstructured->indox)
      Downloading langdetect-1.0.9.tar.gz (981 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 981.5/981.5 kB 50.6 MB/s eta 0:00:00
    etadata (setup.py) ...  unstructured->indox)
      Downloading rapidfuzz-3.9.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.4 MB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.4/3.4 MB 29.4 MB/s eta 0:00:00
     unstructured->indox)
      Downloading backoff-2.2.1-py3-none-any.whl (15 kB)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from unstructured->indox) (4.12.2)
    Collecting unstructured-client (from unstructured->indox)
      Downloading unstructured_client-0.23.8-py3-none-any.whl (40 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 41.0/41.0 kB 2.9 MB/s eta 0:00:00
    ent already satisfied: wrapt in /usr/local/lib/python3.10/dist-packages (from unstructured->indox) (1.14.1)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain->indox) (1.3.1)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain->indox) (23.2.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain->indox) (1.4.1)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain->indox) (6.0.5)
    Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.10/dist-packages (from aiohttp<4.0.0,>=3.8.3->langchain->indox) (1.9.4)
    Collecting marshmallow<4.0.0,>=3.18.0 (from dataclasses-json<0.7,>=0.5.7->langchain-community->indox)
      Downloading marshmallow-3.21.3-py3-none-any.whl (49 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 49.2/49.2 kB 3.0 MB/s eta 0:00:00
     dataclasses-json<0.7,>=0.5.7->langchain-community->indox)
      Downloading typing_inspect-0.9.0-py3-none-any.whl (8.8 kB)
    Requirement already satisfied: anyio in /usr/local/lib/python3.10/dist-packages (from httpx<1,>=0.25.2->langchain-mistralai->indox) (3.7.1)
    Collecting httpcore==1.* (from httpx<1,>=0.25.2->langchain-mistralai->indox)
      Downloading httpcore-1.0.5-py3-none-any.whl (77 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 77.9/77.9 kB 5.1 MB/s eta 0:00:00
    ent already satisfied: sniffio in /usr/local/lib/python3.10/dist-packages (from httpx<1,>=0.25.2->langchain-mistralai->indox) (1.3.1)
    Collecting h11<0.15,>=0.13 (from httpcore==1.*->httpx<1,>=0.25.2->langchain-mistralai->indox)
      Downloading h11-0.14.0-py3-none-any.whl (58 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 58.3/58.3 kB 6.2 MB/s eta 0:00:00
    ent already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.16.4->tokenizers->indox) (3.15.3)
    Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.16.4->tokenizers->indox) (2023.6.0)
    Collecting jsonpointer>=1.9 (from jsonpatch<2.0,>=1.33->langchain-core->indox)
      Downloading jsonpointer-3.0.0-py2.py3-none-any.whl (7.6 kB)
    Collecting orjson<4.0.0,>=3.9.14 (from langsmith<0.2.0,>=0.1.17->langchain->indox)
      Downloading orjson-3.10.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (144 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 145.0/145.0 kB 4.1 MB/s eta 0:00:00
    ent already satisfied: llvmlite<0.42,>=0.41.0dev0 in /usr/local/lib/python3.10/dist-packages (from numba>=0.51.2->umap-learn->indox) (0.41.1)
    Requirement already satisfied: distro<2,>=1.7.0 in /usr/lib/python3/dist-packages (from openai<2.0.0,>=1.32.0->langchain-openai->indox) (1.7.0)
    Requirement already satisfied: annotated-types>=0.4.0 in /usr/local/lib/python3.10/dist-packages (from pydantic<3,>=1->langchain->indox) (0.7.0)
    Requirement already satisfied: pydantic-core==2.18.4 in /usr/local/lib/python3.10/dist-packages (from pydantic<3,>=1->langchain->indox) (2.18.4)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas~=2.0.3->indox) (1.16.0)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn>=0.22->umap-learn->indox) (3.5.0)
    Requirement already satisfied: greenlet!=0.4.17 in /usr/local/lib/python3.10/dist-packages (from SQLAlchemy<3,>=1.4->langchain->indox) (3.0.3)
    Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.10/dist-packages (from beautifulsoup4->unstructured->indox) (2.5)
    Collecting deepdiff>=6.0 (from unstructured-client->unstructured->indox)
      Downloading deepdiff-7.0.1-py3-none-any.whl (80 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 80.8/80.8 kB 7.9 MB/s eta 0:00:00
     unstructured-client->unstructured->indox)
      Downloading jsonpath_python-1.0.6-py3-none-any.whl (7.6 kB)
    Collecting mypy-extensions>=1.0.0 (from unstructured-client->unstructured->indox)
      Downloading mypy_extensions-1.0.0-py3-none-any.whl (4.7 kB)
    Requirement already satisfied: nest-asyncio>=1.6.0 in /usr/local/lib/python3.10/dist-packages (from unstructured-client->unstructured->indox) (1.6.0)
    Collecting pypdf>=4.0 (from unstructured-client->unstructured->indox)
      Downloading pypdf-4.2.0-py3-none-any.whl (290 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 290.4/290.4 kB 28.0 MB/s eta 0:00:00
     unstructured-client->unstructured->indox)
      Downloading requests_toolbelt-1.0.0-py2.py3-none-any.whl (54 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 54.5/54.5 kB 6.1 MB/s eta 0:00:00
    ent already satisfied: exceptiongroup in /usr/local/lib/python3.10/dist-packages (from anyio->httpx<1,>=0.25.2->langchain-mistralai->indox) (1.2.1)
    Collecting ordered-set<4.2.0,>=4.1.0 (from deepdiff>=6.0->unstructured-client->unstructured->indox)
      Downloading ordered_set-4.1.0-py3-none-any.whl (7.6 kB)
    Building wheels for collected packages: utils, latex2markdown, langdetect
      Building wheel for utils (setup.py) ... e=utils-1.0.2-py2.py3-none-any.whl size=13906 sha256=9fe6d4555192f730ae4a4e26e3b9a6851d28fecf4686af0c10654dcdc2237859
      Stored in directory: /root/.cache/pip/wheels/b8/39/f5/9d0ca31dba85773ececf0a7f5469f18810e1c8a8ed9da28ca7
      Building wheel for latex2markdown (setup.py) ... arkdown: filename=latex2markdown-0.2.1-py3-none-any.whl size=8983 sha256=deef9a64953194243b34d5d77fb3cea36ab147b8cd16c10d7fbb4c214278eb57
      Stored in directory: /root/.cache/pip/wheels/48/d0/8a/5009532cfcf3a270e1126c6a46d73ab028737188faf768fbba
      Building wheel for langdetect (setup.py) ... e=langdetect-1.0.9-py3-none-any.whl size=993227 sha256=8560ff85a61a4a8717bc5761ac23954579deeca8028587ec83e5f67a4a99f2c9
      Stored in directory: /root/.cache/pip/wheels/95/03/7d/59ea870c70ce4e5a370638b5462a7711ab78fba2f655d05106
    Successfully built utils latex2markdown langdetect
    Installing collected packages: latex2markdown, filetype, utils, rapidfuzz, python-magic, python-iso639, python-dotenv, PyPDF2, pypdf, orjson, ordered-set, mypy-extensions, marshmallow, langdetect, jsonpointer, jsonpath-python, httpx-sse, h11, emoji, backoff, typing-inspect, tiktoken, requests-toolbelt, jsonpatch, httpcore, deepdiff, pynndescent, langsmith, httpx, dataclasses-json, unstructured-client, umap-learn, openai, langchain-core, unstructured, langgraph, langchain-text-splitters, langchain-openai, langchain-mistralai, langchain, langchain-community, indox
    Successfully installed PyPDF2-3.0.1 backoff-2.2.1 dataclasses-json-0.6.7 deepdiff-7.0.1 emoji-2.12.1 filetype-1.2.0 h11-0.14.0 httpcore-1.0.5 httpx-0.27.0 httpx-sse-0.4.0 indox-0.1.11 jsonpatch-1.33 jsonpath-python-1.0.6 jsonpointer-3.0.0 langchain-0.2.6 langchain-community-0.2.6 langchain-core-0.2.10 langchain-mistralai-0.1.9 langchain-openai-0.1.13 langchain-text-splitters-0.2.2 langdetect-1.0.9 langgraph-0.1.4 langsmith-0.1.82 latex2markdown-0.2.1 marshmallow-3.21.3 mypy-extensions-1.0.0 openai-1.35.7 ordered-set-4.1.0 orjson-3.10.5 pynndescent-0.5.13 pypdf-4.2.0 python-dotenv-1.0.1 python-iso639-2024.4.27 python-magic-0.4.27 rapidfuzz-3.9.3 requests-toolbelt-1.0.0 tiktoken-0.7.0 typing-inspect-0.9.0 umap-learn-0.5.6 unstructured-0.14.9 unstructured-client-0.23.8 utils-1.0.2
    Requirement already satisfied: openai in /usr/local/lib/python3.10/dist-packages (1.35.7)
    Requirement already satisfied: anyio<5,>=3.5.0 in /usr/local/lib/python3.10/dist-packages (from openai) (3.7.1)
    Requirement already satisfied: distro<2,>=1.7.0 in /usr/lib/python3/dist-packages (from openai) (1.7.0)
    Requirement already satisfied: httpx<1,>=0.23.0 in /usr/local/lib/python3.10/dist-packages (from openai) (0.27.0)
    Requirement already satisfied: pydantic<3,>=1.9.0 in /usr/local/lib/python3.10/dist-packages (from openai) (2.7.4)
    Requirement already satisfied: sniffio in /usr/local/lib/python3.10/dist-packages (from openai) (1.3.1)
    Requirement already satisfied: tqdm>4 in /usr/local/lib/python3.10/dist-packages (from openai) (4.66.4)
    Requirement already satisfied: typing-extensions<5,>=4.7 in /usr/local/lib/python3.10/dist-packages (from openai) (4.12.2)
    Requirement already satisfied: idna>=2.8 in /usr/local/lib/python3.10/dist-packages (from anyio<5,>=3.5.0->openai) (3.7)
    Requirement already satisfied: exceptiongroup in /usr/local/lib/python3.10/dist-packages (from anyio<5,>=3.5.0->openai) (1.2.1)
    Requirement already satisfied: certifi in /usr/local/lib/python3.10/dist-packages (from httpx<1,>=0.23.0->openai) (2024.6.2)
    Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.10/dist-packages (from httpx<1,>=0.23.0->openai) (1.0.5)
    Requirement already satisfied: h11<0.15,>=0.13 in /usr/local/lib/python3.10/dist-packages (from httpcore==1.*->httpx<1,>=0.23.0->openai) (0.14.0)
    Requirement already satisfied: annotated-types>=0.4.0 in /usr/local/lib/python3.10/dist-packages (from pydantic<3,>=1.9.0->openai) (0.7.0)
    Requirement already satisfied: pydantic-core==2.18.4 in /usr/local/lib/python3.10/dist-packages (from pydantic<3,>=1.9.0->openai) (2.18.4)
    Collecting chromadb
      Downloading chromadb-0.5.3-py3-none-any.whl (559 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 559.5/559.5 kB 9.1 MB/s eta 0:00:00
    ent already satisfied: build>=1.0.3 in /usr/local/lib/python3.10/dist-packages (from chromadb) (1.2.1)
    Requirement already satisfied: requests>=2.28 in /usr/local/lib/python3.10/dist-packages (from chromadb) (2.31.0)
    Requirement already satisfied: pydantic>=1.9 in /usr/local/lib/python3.10/dist-packages (from chromadb) (2.7.4)
    Collecting chroma-hnswlib==0.7.3 (from chromadb)
      Downloading chroma_hnswlib-0.7.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.4 MB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.4/2.4 MB 45.2 MB/s eta 0:00:00
     chromadb)
      Downloading fastapi-0.111.0-py3-none-any.whl (91 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 92.0/92.0 kB 9.9 MB/s eta 0:00:00
     chromadb)
      Downloading uvicorn-0.30.1-py3-none-any.whl (62 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 62.4/62.4 kB 6.7 MB/s eta 0:00:00
    ent already satisfied: numpy<2.0.0,>=1.22.5 in /usr/local/lib/python3.10/dist-packages (from chromadb) (1.25.2)
    Collecting posthog>=2.4.0 (from chromadb)
      Downloading posthog-3.5.0-py2.py3-none-any.whl (41 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 41.3/41.3 kB 4.3 MB/s eta 0:00:00
    ent already satisfied: typing-extensions>=4.5.0 in /usr/local/lib/python3.10/dist-packages (from chromadb) (4.12.2)
    Collecting onnxruntime>=1.14.1 (from chromadb)
      Downloading onnxruntime-1.18.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (6.8 MB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.8/6.8 MB 81.7 MB/s eta 0:00:00
    etry-api>=1.2.0 (from chromadb)
      Downloading opentelemetry_api-1.25.0-py3-none-any.whl (59 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 59.9/59.9 kB 6.9 MB/s eta 0:00:00
    etry-exporter-otlp-proto-grpc>=1.2.0 (from chromadb)
      Downloading opentelemetry_exporter_otlp_proto_grpc-1.25.0-py3-none-any.whl (18 kB)
    Collecting opentelemetry-instrumentation-fastapi>=0.41b0 (from chromadb)
      Downloading opentelemetry_instrumentation_fastapi-0.46b0-py3-none-any.whl (11 kB)
    Collecting opentelemetry-sdk>=1.2.0 (from chromadb)
      Downloading opentelemetry_sdk-1.25.0-py3-none-any.whl (107 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 107.0/107.0 kB 11.1 MB/s eta 0:00:00
    ent already satisfied: tokenizers>=0.13.2 in /usr/local/lib/python3.10/dist-packages (from chromadb) (0.19.1)
    Collecting pypika>=0.48.9 (from chromadb)
      Downloading PyPika-0.48.9.tar.gz (67 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 67.3/67.3 kB 5.8 MB/s eta 0:00:00
    ents to build wheel ... etadata (pyproject.toml) ... ent already satisfied: tqdm>=4.65.0 in /usr/local/lib/python3.10/dist-packages (from chromadb) (4.66.4)
    Collecting overrides>=7.3.1 (from chromadb)
      Downloading overrides-7.7.0-py3-none-any.whl (17 kB)
    Requirement already satisfied: importlib-resources in /usr/local/lib/python3.10/dist-packages (from chromadb) (6.4.0)
    Requirement already satisfied: grpcio>=1.58.0 in /usr/local/lib/python3.10/dist-packages (from chromadb) (1.64.1)
    Collecting bcrypt>=4.0.1 (from chromadb)
      Downloading bcrypt-4.1.3-cp39-abi3-manylinux_2_28_x86_64.whl (283 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 283.7/283.7 kB 27.6 MB/s eta 0:00:00
    ent already satisfied: typer>=0.9.0 in /usr/local/lib/python3.10/dist-packages (from chromadb) (0.12.3)
    Collecting kubernetes>=28.1.0 (from chromadb)
      Downloading kubernetes-30.1.0-py2.py3-none-any.whl (1.7 MB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.7/1.7 MB 68.0 MB/s eta 0:00:00
    ent already satisfied: tenacity>=8.2.3 in /usr/local/lib/python3.10/dist-packages (from chromadb) (8.4.1)
    Requirement already satisfied: PyYAML>=6.0.0 in /usr/local/lib/python3.10/dist-packages (from chromadb) (6.0.1)
    Collecting mmh3>=4.0.1 (from chromadb)
      Downloading mmh3-4.1.0-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (67 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 67.6/67.6 kB 1.6 MB/s eta 0:00:00
    ent already satisfied: orjson>=3.9.12 in /usr/local/lib/python3.10/dist-packages (from chromadb) (3.10.5)
    Requirement already satisfied: httpx>=0.27.0 in /usr/local/lib/python3.10/dist-packages (from chromadb) (0.27.0)
    Requirement already satisfied: packaging>=19.1 in /usr/local/lib/python3.10/dist-packages (from build>=1.0.3->chromadb) (24.1)
    Requirement already satisfied: pyproject_hooks in /usr/local/lib/python3.10/dist-packages (from build>=1.0.3->chromadb) (1.1.0)
    Requirement already satisfied: tomli>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from build>=1.0.3->chromadb) (2.0.1)
    Collecting starlette<0.38.0,>=0.37.2 (from fastapi>=0.95.2->chromadb)
      Downloading starlette-0.37.2-py3-none-any.whl (71 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 71.9/71.9 kB 7.1 MB/s eta 0:00:00
     fastapi>=0.95.2->chromadb)
      Downloading fastapi_cli-0.0.4-py3-none-any.whl (9.5 kB)
    Requirement already satisfied: jinja2>=2.11.2 in /usr/local/lib/python3.10/dist-packages (from fastapi>=0.95.2->chromadb) (3.1.4)
    Collecting python-multipart>=0.0.7 (from fastapi>=0.95.2->chromadb)
      Downloading python_multipart-0.0.9-py3-none-any.whl (22 kB)
    Collecting ujson!=4.0.2,!=4.1.0,!=4.2.0,!=4.3.0,!=5.0.0,!=5.1.0,>=4.0.1 (from fastapi>=0.95.2->chromadb)
      Downloading ujson-5.10.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (53 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 53.6/53.6 kB 5.4 MB/s eta 0:00:00
    ail_validator>=2.0.0 (from fastapi>=0.95.2->chromadb)
      Downloading email_validator-2.2.0-py3-none-any.whl (33 kB)
    Requirement already satisfied: anyio in /usr/local/lib/python3.10/dist-packages (from httpx>=0.27.0->chromadb) (3.7.1)
    Requirement already satisfied: certifi in /usr/local/lib/python3.10/dist-packages (from httpx>=0.27.0->chromadb) (2024.6.2)
    Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.10/dist-packages (from httpx>=0.27.0->chromadb) (1.0.5)
    Requirement already satisfied: idna in /usr/local/lib/python3.10/dist-packages (from httpx>=0.27.0->chromadb) (3.7)
    Requirement already satisfied: sniffio in /usr/local/lib/python3.10/dist-packages (from httpx>=0.27.0->chromadb) (1.3.1)
    Requirement already satisfied: h11<0.15,>=0.13 in /usr/local/lib/python3.10/dist-packages (from httpcore==1.*->httpx>=0.27.0->chromadb) (0.14.0)
    Requirement already satisfied: six>=1.9.0 in /usr/local/lib/python3.10/dist-packages (from kubernetes>=28.1.0->chromadb) (1.16.0)
    Requirement already satisfied: python-dateutil>=2.5.3 in /usr/local/lib/python3.10/dist-packages (from kubernetes>=28.1.0->chromadb) (2.8.2)
    Requirement already satisfied: google-auth>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from kubernetes>=28.1.0->chromadb) (2.27.0)
    Requirement already satisfied: websocket-client!=0.40.0,!=0.41.*,!=0.42.*,>=0.32.0 in /usr/local/lib/python3.10/dist-packages (from kubernetes>=28.1.0->chromadb) (1.8.0)
    Requirement already satisfied: requests-oauthlib in /usr/local/lib/python3.10/dist-packages (from kubernetes>=28.1.0->chromadb) (1.3.1)
    Requirement already satisfied: oauthlib>=3.2.2 in /usr/local/lib/python3.10/dist-packages (from kubernetes>=28.1.0->chromadb) (3.2.2)
    Requirement already satisfied: urllib3>=1.24.2 in /usr/local/lib/python3.10/dist-packages (from kubernetes>=28.1.0->chromadb) (2.0.7)
    Collecting coloredlogs (from onnxruntime>=1.14.1->chromadb)
      Downloading coloredlogs-15.0.1-py2.py3-none-any.whl (46 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 46.0/46.0 kB 4.4 MB/s eta 0:00:00
    ent already satisfied: flatbuffers in /usr/local/lib/python3.10/dist-packages (from onnxruntime>=1.14.1->chromadb) (24.3.25)
    Requirement already satisfied: protobuf in /usr/local/lib/python3.10/dist-packages (from onnxruntime>=1.14.1->chromadb) (3.20.3)
    Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from onnxruntime>=1.14.1->chromadb) (1.12.1)
    Collecting deprecated>=1.2.6 (from opentelemetry-api>=1.2.0->chromadb)
      Downloading Deprecated-1.2.14-py2.py3-none-any.whl (9.6 kB)
    Collecting importlib-metadata<=7.1,>=6.0 (from opentelemetry-api>=1.2.0->chromadb)
      Downloading importlib_metadata-7.1.0-py3-none-any.whl (24 kB)
    Requirement already satisfied: googleapis-common-protos~=1.52 in /usr/local/lib/python3.10/dist-packages (from opentelemetry-exporter-otlp-proto-grpc>=1.2.0->chromadb) (1.63.1)
    Collecting opentelemetry-exporter-otlp-proto-common==1.25.0 (from opentelemetry-exporter-otlp-proto-grpc>=1.2.0->chromadb)
      Downloading opentelemetry_exporter_otlp_proto_common-1.25.0-py3-none-any.whl (17 kB)
    Collecting opentelemetry-proto==1.25.0 (from opentelemetry-exporter-otlp-proto-grpc>=1.2.0->chromadb)
      Downloading opentelemetry_proto-1.25.0-py3-none-any.whl (52 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 52.5/52.5 kB 5.6 MB/s eta 0:00:00
    etry-instrumentation-asgi==0.46b0 (from opentelemetry-instrumentation-fastapi>=0.41b0->chromadb)
      Downloading opentelemetry_instrumentation_asgi-0.46b0-py3-none-any.whl (14 kB)
    Collecting opentelemetry-instrumentation==0.46b0 (from opentelemetry-instrumentation-fastapi>=0.41b0->chromadb)
      Downloading opentelemetry_instrumentation-0.46b0-py3-none-any.whl (29 kB)
    Collecting opentelemetry-semantic-conventions==0.46b0 (from opentelemetry-instrumentation-fastapi>=0.41b0->chromadb)
      Downloading opentelemetry_semantic_conventions-0.46b0-py3-none-any.whl (130 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 130.5/130.5 kB 14.0 MB/s eta 0:00:00
    etry-util-http==0.46b0 (from opentelemetry-instrumentation-fastapi>=0.41b0->chromadb)
      Downloading opentelemetry_util_http-0.46b0-py3-none-any.whl (6.9 kB)
    Requirement already satisfied: setuptools>=16.0 in /usr/local/lib/python3.10/dist-packages (from opentelemetry-instrumentation==0.46b0->opentelemetry-instrumentation-fastapi>=0.41b0->chromadb) (67.7.2)
    Requirement already satisfied: wrapt<2.0.0,>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from opentelemetry-instrumentation==0.46b0->opentelemetry-instrumentation-fastapi>=0.41b0->chromadb) (1.14.1)
    Collecting asgiref~=3.0 (from opentelemetry-instrumentation-asgi==0.46b0->opentelemetry-instrumentation-fastapi>=0.41b0->chromadb)
      Downloading asgiref-3.8.1-py3-none-any.whl (23 kB)
    Collecting monotonic>=1.5 (from posthog>=2.4.0->chromadb)
      Downloading monotonic-1.6-py2.py3-none-any.whl (8.2 kB)
    Requirement already satisfied: backoff>=1.10.0 in /usr/local/lib/python3.10/dist-packages (from posthog>=2.4.0->chromadb) (2.2.1)
    Requirement already satisfied: annotated-types>=0.4.0 in /usr/local/lib/python3.10/dist-packages (from pydantic>=1.9->chromadb) (0.7.0)
    Requirement already satisfied: pydantic-core==2.18.4 in /usr/local/lib/python3.10/dist-packages (from pydantic>=1.9->chromadb) (2.18.4)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.28->chromadb) (3.3.2)
    Requirement already satisfied: huggingface-hub<1.0,>=0.16.4 in /usr/local/lib/python3.10/dist-packages (from tokenizers>=0.13.2->chromadb) (0.23.4)
    Requirement already satisfied: click>=8.0.0 in /usr/local/lib/python3.10/dist-packages (from typer>=0.9.0->chromadb) (8.1.7)
    Requirement already satisfied: shellingham>=1.3.0 in /usr/local/lib/python3.10/dist-packages (from typer>=0.9.0->chromadb) (1.5.4)
    Requirement already satisfied: rich>=10.11.0 in /usr/local/lib/python3.10/dist-packages (from typer>=0.9.0->chromadb) (13.7.1)
    Collecting httptools>=0.5.0 (from uvicorn[standard]>=0.18.3->chromadb)
      Downloading httptools-0.6.1-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (341 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 341.4/341.4 kB 31.3 MB/s eta 0:00:00
    ent already satisfied: python-dotenv>=0.13 in /usr/local/lib/python3.10/dist-packages (from uvicorn[standard]>=0.18.3->chromadb) (1.0.1)
    Collecting uvloop!=0.15.0,!=0.15.1,>=0.14.0 (from uvicorn[standard]>=0.18.3->chromadb)
      Downloading uvloop-0.19.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.4 MB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.4/3.4 MB 39.6 MB/s eta 0:00:00
     uvicorn[standard]>=0.18.3->chromadb)
      Downloading watchfiles-0.22.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.2 MB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 48.5 MB/s eta 0:00:00
     uvicorn[standard]>=0.18.3->chromadb)
      Downloading websockets-12.0-cp310-cp310-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_17_x86_64.manylinux2014_x86_64.whl (130 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 130.2/130.2 kB 14.7 MB/s eta 0:00:00
     email_validator>=2.0.0->fastapi>=0.95.2->chromadb)
      Downloading dnspython-2.6.1-py3-none-any.whl (307 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 307.7/307.7 kB 26.4 MB/s eta 0:00:00
    ent already satisfied: cachetools<6.0,>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from google-auth>=1.0.1->kubernetes>=28.1.0->chromadb) (5.3.3)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.10/dist-packages (from google-auth>=1.0.1->kubernetes>=28.1.0->chromadb) (0.4.0)
    Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.10/dist-packages (from google-auth>=1.0.1->kubernetes>=28.1.0->chromadb) (4.9)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.16.4->tokenizers>=0.13.2->chromadb) (3.15.3)
    Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.10/dist-packages (from huggingface-hub<1.0,>=0.16.4->tokenizers>=0.13.2->chromadb) (2023.6.0)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.10/dist-packages (from importlib-metadata<=7.1,>=6.0->opentelemetry-api>=1.2.0->chromadb) (3.19.2)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2>=2.11.2->fastapi>=0.95.2->chromadb) (2.1.5)
    Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich>=10.11.0->typer>=0.9.0->chromadb) (3.0.0)
    Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich>=10.11.0->typer>=0.9.0->chromadb) (2.16.1)
    Requirement already satisfied: exceptiongroup in /usr/local/lib/python3.10/dist-packages (from anyio->httpx>=0.27.0->chromadb) (1.2.1)
    Collecting humanfriendly>=9.1 (from coloredlogs->onnxruntime>=1.14.1->chromadb)
      Downloading humanfriendly-10.0-py2.py3-none-any.whl (86 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 86.8/86.8 kB 10.6 MB/s eta 0:00:00
    ent already satisfied: mpmath<1.4.0,>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from sympy->onnxruntime>=1.14.1->chromadb) (1.3.0)
    Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich>=10.11.0->typer>=0.9.0->chromadb) (0.1.2)
    Requirement already satisfied: pyasn1<0.7.0,>=0.4.6 in /usr/local/lib/python3.10/dist-packages (from pyasn1-modules>=0.2.1->google-auth>=1.0.1->kubernetes>=28.1.0->chromadb) (0.6.0)
    Building wheels for collected packages: pypika
      Building wheel for pypika (pyproject.toml) ... e=PyPika-0.48.9-py2.py3-none-any.whl size=53726 sha256=7d704afd4589850cf5ea3f7c5d2129673ed73b78f268d3a6837e5fe024134ca4
      Stored in directory: /root/.cache/pip/wheels/e1/26/51/d0bffb3d2fd82256676d7ad3003faea3bd6dddc9577af665f4
    Successfully built pypika
    Installing collected packages: pypika, monotonic, mmh3, websockets, uvloop, uvicorn, ujson, python-multipart, overrides, opentelemetry-util-http, opentelemetry-proto, importlib-metadata, humanfriendly, httptools, dnspython, deprecated, chroma-hnswlib, bcrypt, asgiref, watchfiles, starlette, posthog, opentelemetry-exporter-otlp-proto-common, opentelemetry-api, email_validator, coloredlogs, opentelemetry-semantic-conventions, opentelemetry-instrumentation, onnxruntime, kubernetes, opentelemetry-sdk, opentelemetry-instrumentation-asgi, fastapi-cli, opentelemetry-instrumentation-fastapi, opentelemetry-exporter-otlp-proto-grpc, fastapi, chromadb
      Attempting uninstall: importlib-metadata
        Found existing installation: importlib_metadata 7.2.0
        Uninstalling importlib_metadata-7.2.0:
          Successfully uninstalled importlib_metadata-7.2.0
    Successfully installed asgiref-3.8.1 bcrypt-4.1.3 chroma-hnswlib-0.7.3 chromadb-0.5.3 coloredlogs-15.0.1 deprecated-1.2.14 dnspython-2.6.1 email_validator-2.2.0 fastapi-0.111.0 fastapi-cli-0.0.4 httptools-0.6.1 humanfriendly-10.0 importlib-metadata-7.1.0 kubernetes-30.1.0 mmh3-4.1.0 monotonic-1.6 onnxruntime-1.18.1 opentelemetry-api-1.25.0 opentelemetry-exporter-otlp-proto-common-1.25.0 opentelemetry-exporter-otlp-proto-grpc-1.25.0 opentelemetry-instrumentation-0.46b0 opentelemetry-instrumentation-asgi-0.46b0 opentelemetry-instrumentation-fastapi-0.46b0 opentelemetry-proto-1.25.0 opentelemetry-sdk-1.25.0 opentelemetry-semantic-conventions-0.46b0 opentelemetry-util-http-0.46b0 overrides-7.7.0 posthog-3.5.0 pypika-0.48.9 python-multipart-0.0.9 starlette-0.37.2 ujson-5.10.0 uvicorn-0.30.1 uvloop-0.19.0 watchfiles-0.22.0 websockets-12.0
:::
:::

::: {#ec52f0c0a7c8f592 .cell .code execution_count="2" ExecuteTime="{\"end_time\":\"2024-06-09T10:26:51.269942Z\",\"start_time\":\"2024-06-09T10:26:51.252915Z\"}" id="ec52f0c0a7c8f592"}
``` python
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
```
:::

::: {#7f71c391 .cell .markdown id="7f71c391"}
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
