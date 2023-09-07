import os
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
from configs.model_config import (
    embedding_model_dict,
    KB_ROOT_PATH,
    CHUNK_SIZE,
    OVERLAP_SIZE,
    ZH_TITLE_ENHANCE
)
from functools import lru_cache
import importlib
from text_splitter import zh_title_enhance
import langchain.document_loaders
from langchain.docstore.document import Document
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Callable, Dict, Optional, Tuple, Generator


# make HuggingFaceEmbeddings hashable
def _embeddings_hash(self):
    if isinstance(self, HuggingFaceEmbeddings):
        return hash(self.model_name)
    elif isinstance(self, HuggingFaceBgeEmbeddings):
        return hash(self.model_name)
    elif isinstance(self, OpenAIEmbeddings):
        return hash(self.model)


HuggingFaceEmbeddings.__hash__ = _embeddings_hash
OpenAIEmbeddings.__hash__ = _embeddings_hash
HuggingFaceBgeEmbeddings.__hash__ = _embeddings_hash


def validate_kb_name(knowledge_base_id: str) -> bool:
    # 检查是否包含预期外的字符或路径攻击关键字
    if "../" in knowledge_base_id:
        return False
    return True


def get_kb_path(knowledge_base_name: str):
    return os.path.join(KB_ROOT_PATH, knowledge_base_name)


def get_doc_path(knowledge_base_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "content")


def get_vs_path(knowledge_base_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "vector_store")


def get_file_path(knowledge_base_name: str, doc_name: str):
    return os.path.join(get_doc_path(knowledge_base_name), doc_name)


def list_kbs_from_folder():
    return [f for f in os.listdir(KB_ROOT_PATH)
            if os.path.isdir(os.path.join(KB_ROOT_PATH, f))]


def list_files_from_folder(kb_name: str):
    doc_path = get_doc_path(kb_name)
    return [file for file in os.listdir(doc_path)
            if os.path.isfile(os.path.join(doc_path, file))]


@lru_cache(1)
def load_embeddings(model: str, device: str):
    if model == "text-embedding-ada-002":  # openai text-embedding-ada-002
        embeddings = OpenAIEmbeddings(openai_api_key=embedding_model_dict[model], chunk_size=CHUNK_SIZE)
    elif 'bge-' in model:
        embeddings = HuggingFaceBgeEmbeddings(model_name=embedding_model_dict[model],
                                              model_kwargs={'device': device},
                                              query_instruction="为这个句子生成表示以用于检索相关文章：")
        if model == "bge-large-zh-noinstruct":  # bge large -noinstruct embedding
            embeddings.query_instruction = ""
    else:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[model], model_kwargs={'device': device})
    return embeddings


LOADER_DICT = {"UnstructuredHTMLLoader": ['.html'],
               "UnstructuredMarkdownLoader": ['.md'],
               "CustomJSONLoader": [".json"],
               "CSVLoader": [".csv"],
               "RapidOCRPDFLoader": [".pdf"],
               "RapidOCRLoader": ['.png', '.jpg', '.jpeg', '.bmp'],
               "UnstructuredFileLoader": ['.eml', '.msg', '.rst',
                                          '.rtf', '.txt', '.xml',
                                          '.doc', '.docx', '.epub', '.odt',
                                          '.ppt', '.pptx', '.tsv'],  # '.xlsx'
               }
SUPPORTED_EXTS = [ext for sublist in LOADER_DICT.values() for ext in sublist]


class CustomJSONLoader(langchain.document_loaders.JSONLoader):
    '''
    langchain的JSONLoader需要jq，在win上使用不便，进行替代。
    '''

    def __init__(
            self,
            file_path: Union[str, Path],
            content_key: Optional[str] = None,
            metadata_func: Optional[Callable[[Dict, Dict], Dict]] = None,
            text_content: bool = True,
            json_lines: bool = False,
    ):
        """Initialize the JSONLoader.

        Args:
            file_path (Union[str, Path]): The path to the JSON or JSON Lines file.
            content_key (str): The key to use to extract the content from the JSON if
                results to a list of objects (dict).
            metadata_func (Callable[Dict, Dict]): A function that takes in the JSON
                object extracted by the jq_schema and the default metadata and returns
                a dict of the updated metadata.
            text_content (bool): Boolean flag to indicate whether the content is in
                string format, default to True.
            json_lines (bool): Boolean flag to indicate whether the input is in
                JSON Lines format.
        """
        self.file_path = Path(file_path).resolve()
        self._content_key = content_key
        self._metadata_func = metadata_func
        self._text_content = text_content
        self._json_lines = json_lines

    # TODO: langchain's JSONLoader.load has a encoding bug, raise gbk encoding error on windows.
    # This is a workaround for langchain==0.0.266. I have make a pr(#9785) to langchain, it should be deleted after langchain upgraded.
    def load(self) -> List[Document]:
        """Load and return documents from the JSON file."""
        docs: List[Document] = []
        if self._json_lines:
            with self.file_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._parse(line, docs)
        else:
            self._parse(self.file_path.read_text(encoding="utf-8"), docs)
        return docs

    def _parse(self, content: str, docs: List[Document]) -> None:
        """Convert given content to documents."""
        data = json.loads(content)

        # Perform some validation
        # This is not a perfect validation, but it should catch most cases
        # and prevent the user from getting a cryptic error later on.
        if self._content_key is not None:
            self._validate_content_key(data)

        for i, sample in enumerate(data, len(docs) + 1):
            metadata = dict(
                source=str(self.file_path),
                seq_num=i,
            )
            text = self._get_text(sample=sample, metadata=metadata)
            docs.append(Document(page_content=text, metadata=metadata))


langchain.document_loaders.CustomJSONLoader = CustomJSONLoader


def get_LoaderClass(file_extension):
    for LoaderClass, extensions in LOADER_DICT.items():
        if file_extension in extensions:
            return LoaderClass


class KnowledgeFile:
    def __init__(
            self,
            filename: str,
            knowledge_base_name: str
    ):
        self.kb_name = knowledge_base_name
        self.filename = filename
        self.ext = os.path.splitext(filename)[-1].lower()
        if self.ext not in SUPPORTED_EXTS:
            raise ValueError(f"暂未支持的文件格式 {self.ext}")
        self.filepath = get_file_path(knowledge_base_name, filename)
        self.docs = None
        self.document_loader_name = get_LoaderClass(self.ext)

        # TODO: 增加依据文件格式匹配text_splitter
        self.text_splitter_name = None

    # 文件转txt并切割chunk
    def file2text(self, using_zh_title_enhance=ZH_TITLE_ENHANCE, refresh: bool = False):
        if self.docs is not None and not refresh:
            return self.docs
        # UnstructuredFileLoader used for E:\AI\Langchain-Chatchat\knowledge_base\langchain\content\8月报.txt
        print(f"{self.document_loader_name} used for {self.filepath}")
        try:
            if self.document_loader_name in ["RapidOCRPDFLoader", "RapidOCRLoader"]:
                document_loaders_module = importlib.import_module('document_loaders')
            else:  # 疑问？
                document_loaders_module = importlib.import_module('langchain.document_loaders')
            DocumentLoader = getattr(document_loaders_module, self.document_loader_name)
        except Exception as e:
            print(e)
            document_loaders_module = importlib.import_module('langchain.document_loaders')
            DocumentLoader = getattr(document_loaders_module, "UnstructuredFileLoader")
        if self.document_loader_name == "UnstructuredFileLoader":
            loader = DocumentLoader(self.filepath, autodetect_encoding=True)
        elif self.document_loader_name == "CSVLoader":
            loader = DocumentLoader(self.filepath, encoding="utf-8")
        elif self.document_loader_name == "JSONLoader":
            loader = DocumentLoader(self.filepath, jq_schema=".", text_content=False)
        elif self.document_loader_name == "CustomJSONLoader":
            loader = DocumentLoader(self.filepath, text_content=False)
        elif self.document_loader_name == "UnstructuredMarkdownLoader":
            loader = DocumentLoader(self.filepath, mode="elements")
        elif self.document_loader_name == "UnstructuredHTMLLoader":
            loader = DocumentLoader(self.filepath, mode="elements")
        else:
            loader = DocumentLoader(self.filepath)

        if self.ext in ".csv":
            docs = loader.load()
        else:
            try:  # 如果文本切割名称为null
                if self.text_splitter_name is None:
                    text_splitter_module = importlib.import_module('langchain.text_splitter') # pip install spacy -i https://pypi.tuna.tsinghua.edu.cn/simple spacy的实体抽取工具
                    TextSplitter = getattr(text_splitter_module, "SpacyTextSplitter")
                    text_splitter = TextSplitter(
                        pipeline="zh_core_web_sm",
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=OVERLAP_SIZE,
                    )
                    self.text_splitter_name = "SpacyTextSplitter"
                else:
                    text_splitter_module = importlib.import_module('langchain.text_splitter')
                    TextSplitter = getattr(text_splitter_module, self.text_splitter_name)
                    text_splitter = TextSplitter(
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=OVERLAP_SIZE)
            except Exception as e:  # spacy不存在的话，就走到异常处理
                print(e)
                text_splitter_module = importlib.import_module('langchain.text_splitter')
                TextSplitter = getattr(text_splitter_module, "RecursiveCharacterTextSplitter")
                text_splitter = TextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=OVERLAP_SIZE,
                )
            # 这里是切割的逻辑，按照chunksize切割，生成数组docs
            docs = loader.load_and_split(text_splitter)  # docs =  [Document(page_content='本月已上线：\n\n功能：\n\n1、文档大小版本功能\n\n2、新增eb文档列表组件支持自定义字段功能\n\n3、登录前门户元素\n\n4、统一组件能力\n\n\n\n模块调用时会自动判断文件类型打开对应预览编辑组件\n\n5、刘志刚关于业务工具集改造需求（业务工具集 UI 设计皮肤库改版(新增UI项目皮肤库、市场设计物料）\n\n6、新增支持列排序，调整宽度，支持管理员自定义列同步所有人，字段缺失和排序\n\n优化及修复：\n\n1、修复开启版本后文档自定义字段保存可能失败的问题', metadata={'source': 'E:\\AI\\Langchain-Chatchat\\knowledge_base\\langchain\\content\\8月报.txt'}), Document(page_content='优化及修复：\n\n1、修复开启版本后文档自定义字段保存可能失败的问题\n\n2、优化将申请打印次数功能调用流程rpc接口超时时间问题\n\n3、修复列表视图下文档列表权限接口/api/doc/documents/documentListPermissions包含敏感词时报错的问题\n\n4、项目文档详情、文档详情另存为优化\n\n5、修复个人知识库开启了主附件打开但别人打开时未主附件打开的问题\n\n6、解决审批状态的文档里的附件在流程的事项关联中打开无权限查看的问题', metadata={'source': 'E:\\AI\\Langchain-Chatchat\\knowledge_base\\langchain\\content\\8月报.txt'}), Document(page_content='6、解决审批状态的文档里的附件在流程的事项关联中打开无权限查看的问题\n\n9、修复能够查看共享项目中的文档但添加共享却提示无权限的问题\n\n7、优化调整单个文档引入虚拟知识库不控权批量导入才控制权限的功能\n\n8、优化新增外部分享页面\n\n\n\n保存到我的文档按钮控制的功能\n\n9、解决绑定了手机号时，文档外部分享依旧提示未绑定手机号的问题\n\n10、解决流程存为文档后，生成文档的反馈更新时间字段没有值的问题\n\n11、优化文档详情另存为功能\n\n12、优化文档浏览框支持过滤非正常、归档状态的文档的功能', metadata={'source': 'E:\\AI\\Langchain-Chatchat\\knowledge_base\\langchain\\content\\8月报.txt'}), Document(page_content='11、优化文档详情另存为功能\n\n12、优化文档浏览框支持过滤非正常、归档状态的文档的功能\n\n13、修复协同文档上传生成的文档修改标题后下载正文还是取的文件标题的问题\n\n15、解决拥有编辑权限时，我的订阅页面右键操作菜单中不需要展示编辑按钮的问题\n\n16、解决知识订阅\n\n\n\n可批准订阅页面列表前面的序号显示错乱的问题\n\n17、解决文件夹上开启禁止事项转发，协同文档中的协同转发、只读转发菜单未同步受控的问题', metadata={'source': 'E:\\AI\\Langchain-Chatchat\\knowledge_base\\langchain\\content\\8月报.txt'}), Document(page_content='17、解决文件夹上开启禁止事项转发，协同文档中的协同转发、只读转发菜单未同步受控的问题\n\n18、解决修改office文档正文后，只更新了更新时间字段，没有更新更新/反馈时间字段，导致按照更新/反馈时间字段排序不对的问题\n\n19、解决高级搜索\n\n\n\n搜索条件多语言问题\n\n20、解决html文档正文中插入视频，外部分享时视频不能播放的问题\n\n21、解决查询所有下级文件夹接口报错的问题\n\n22、修复获取系统知识库\n\n\n\n项目文件夹的下级项目失败的问题', metadata={'source': 'E:\\AI\\Langchain-Chatchat\\knowledge_base\\langchain\\content\\8月报.txt'}), Document(page_content='22、修复获取系统知识库\n\n\n\n项目文件夹的下级项目失败的问题\n\n23、协助其他模块修改标签中不正式的提示语内容\n\n专项相关进度：\n\n1、云盘\n\n\n\n本地文件同步功能--\n\n\n\n（测试进度50%，预计9月上线）\n\n2、外部分享支持分享文件夹、设置权限、查看外部日志--\n\n\n\n（进度40%）\n\n3、知识积分--\n\n\n\n（内部需求评审拆分完成）\n\n4、文控管理--\n\n\n\n（需求设计中）\n\n5、断点续传、秒传--\n\n\n\n（进度40%）\n\n6、知识问答--\n\n\n\n（需求评审中）\n\n本月重点开发：', metadata={'source': 'E:\\AI\\Langchain-Chatchat\\knowledge_base\\langchain\\content\\8月报.txt'}), Document(page_content='（进度40%）\n\n6、知识问答--\n\n\n\n（需求评审中）\n\n本月重点开发：\n\n内容服务：\n\n1、E10支持云盘功能（投入测试中，测试进度60%）\n\n2、多身份创建文档功能（本周暂无进度，需求暂缓）\n\n3、新增全部知识新增文件夹树、新增新建文档菜单、新增文档相关自定义设置（进度90%，等待前端对接）\n\n4、会议模块同屏需求\n\n\n\n主讲人切换文档，参与者需要在页面支持提示文档已切换（需求已经提）\n\n5、智能知识地图需求', metadata={'source': 'E:\\AI\\Langchain-Chatchat\\knowledge_base\\langchain\\content\\8月报.txt'}), Document(page_content='主讲人切换文档，参与者需要在页面支持提示文档已切换（需求已经提）\n\n5、智能知识地图需求\n\n\n\nhtml文档编辑支持生成智能标题，大纲，并形成推荐知识点（此块能力用于EB知识地图中，新需求智能文档助手评审中）\n\n6、文件夹打包下载需求，支持文件夹下级所有的文件按照目录层级结构打包下载（进度90%）\n\n7、插件五期\n\n\n\n关于整合公文的插件预览设置，其中包括金格等，统一由知识提供插件能力（进度75%，需求先暂缓不紧急）', metadata={'source': 'E:\\AI\\Langchain-Chatchat\\knowledge_base\\langchain\\content\\8月报.txt'}), Document(page_content='文件存储： 1、E10支持动态存储规则，支持云私多实例存储（进度90%） 2、云盘三期-文件采集-文件服务提供相关接口开发完成（接口已经提供，对接中，参考云盘进度） 3、断点续传web端开发（进度35%） 4、附件单页加入敏感词检测按钮（进度50%，本周暂无进度） 6、office正文版本开发（进度80%） 7、协同文档保存敏感词检测逻辑改造（开发完成，前后端联调中） 8、中建电子商务存储对接第三方存储系统（交付客户现场，客户测试中） 9、cdn静态头像文件方案确认（进度30%）', metadata={'source': 'E:\\AI\\Langchain-Chatchat\\knowledge_base\\langchain\\content\\8月报.txt'}), Document(page_content='9、cdn静态头像文件方案确认（进度30%） 10、和IM文件不生成文档对接鉴权方案-E9迁移的全部文件（a. 传递鉴权字符串的就用鉴权字符串回到IM鉴权；c. 没传递的如果 docid 是 \xa0A（-1） \xa0的就不鉴权、B （-2）直接没权限、C （正常文档ID）就通过文档鉴权; c. 等IM客户端和文件这边都升级之后 IM这边上传的文件，docid 要么为 -2 要么就是正常docid）（等EM开始） 11、评估对接知网在线编辑需求（期望方案确认后开始对接）', metadata={'source': 'E:\\AI\\Langchain-Chatchat\\knowledge_base\\langchain\\content\\8月报.txt'}), Document(page_content='要么就是正常docid）（等EM开始） 11、评估对接知网在线编辑需求（期望方案确认后开始对接） 12、文件服务/papi接口安全漏洞改造（进度80%） 13、富文本不生成文档的逻辑调整（已提测） 14、国电跨系统预览下载群聊文件方案开发（进度20%） 15、去除配置文件中内外网ip的配置逻辑，改成自动获取ip或者域名（改造ET云逻辑）', metadata={'source': 'E:\\AI\\Langchain-Chatchat\\knowledge_base\\langchain\\content\\8月报.txt'}), Document(page_content='下月计划上线：\n\n1、云盘\n\n\n\n本地文件同步功能\n\n2、智能AI文档助手\n\n\n\n生成大纲，标题等\n\n3、AI接入大模型\n\n\n\n检索本地知识库能力，并能依据垂直领域生成文本\n\n4、知识积分一期实现\n\n专项计划相关进度：\n\n1、云盘\n\n\n\n本地文件同步功能--\n\n\n\n（完成上线）\n\n2、外部分享支持分享文件夹、设置权限、查看外部日志--\n\n\n\n（预计提测）\n\n3、知识积分--\n\n\n\n（预计积分一期实现，完整实现在11月前）\n\n4、文控管理--\n\n\n\n（计划分配任务）\n\n5、断点续传、秒传--', metadata={'source': 'E:\\AI\\Langchain-Chatchat\\knowledge_base\\langchain\\content\\8月报.txt'}), Document(page_content='4、文控管理--\n\n\n\n（计划分配任务）\n\n5、断点续传、秒传--\n\n\n\n（预计进度80%）\n\n6、知识问答--\n\n\n\n（预计初版提测）\n\n下月重点开发：\n\n内容服务：\n\n1、E10支持云盘功能（预计上线）\n\n2、多身份创建文档功能（本周暂无进度，需求暂缓）\n\n3、新增全部知识新增文件夹树、新增新建文档菜单、新增文档相关自定义设置（计划提测）\n\n4、会议模块同屏需求\n\n\n\n主讲人切换文档，参与者需要在页面支持提示文档已切换（计划提测）\n\n5、智能知识地图需求', metadata={'source': 'E:\\AI\\Langchain-Chatchat\\knowledge_base\\langchain\\content\\8月报.txt'}), Document(page_content='主讲人切换文档，参与者需要在页面支持提示文档已切换（计划提测）\n\n5、智能知识地图需求\n\n\n\nhtml文档编辑支持生成智能标题，大纲，并形成推荐知识点（预计上线）\n\n6、文件夹打包下载需求，支持文件夹下级所有的文件按照目录层级结构打包下载（预计上线）\n\n7、插件五期\n\n\n\n关于整合公文的插件预览设置，其中包括金格等，统一由知识提供插件能力（预计提测）', metadata={'source': 'E:\\AI\\Langchain-Chatchat\\knowledge_base\\langchain\\content\\8月报.txt'}), Document(page_content='文件存储： 1、E10支持动态存储规则，支持云私多实例存储（计划上线） 2、云盘三期-文件采集-文件服务提供相关接口开发完成（预计上线） 3、断点续传web端开发（预计进度80%） 4、附件单页加入敏感词检测按钮（预计进度70%） 6、office正文版本开发（预计进度90%） 7、协同文档保存敏感词检测逻辑改造（开发完成，前后端联调中） 8、中建电子商务存储对接第三方存储系统（交付客户现场，客户测试中） 9、cdn静态头像文件方案确认（预计进度80%）', metadata={'source': 'E:\\AI\\Langchain-Chatchat\\knowledge_base\\langchain\\content\\8月报.txt'}), Document(page_content='9、cdn静态头像文件方案确认（预计进度80%） 10、和IM文件不生成文档对接鉴权方案-E9迁移的全部文件（a. 传递鉴权字符串的就用鉴权字符串回到IM鉴权；c. 没传递的如果 docid 是 \xa0A（-1） \xa0的就不鉴权、B （-2）直接没权限、C （正常文档ID）就通过文档鉴权; c. 等IM客户端和文件这边都升级之后 IM这边上传的文件，docid 要么为 -2 要么就是正常docid）（等EM开始） 11、评估对接知网在线编辑需求（期望方案确认后开始对接）', metadata={'source': 'E:\\AI\\Langchain-Chatchat\\knowledge_base\\langchain\\content\\8月报.txt'}), Document(page_content='要么就是正常docid）（等EM开始） 11、评估对接知网在线编辑需求（期望方案确认后开始对接） 12、文件服务/papi接口安全漏洞改造（预计提测） 13、富文本不生成文档的逻辑调整（预计上线） 14、国电跨系统预览下载群聊文件方案开发（预计提测） 15、去除配置文件中内外网ip的配置逻辑，改成自动获取ip或者域名（计划上线）', metadata={'source': 'E:\\AI\\Langchain-Chatchat\\knowledge_base\\langchain\\content\\8月报.txt'})]
        print(docs[0])
        if using_zh_title_enhance:  # 是否使用中文标题增强
            docs = zh_title_enhance(docs)
        self.docs = docs
        return docs

    def get_mtime(self):
        return os.path.getmtime(self.filepath)  # os.path.getmtime返回一个UTC时间戳

    def get_size(self):
        return os.path.getsize(self.filepath)


def run_in_thread_pool(
        func: Callable,
        params: List[Dict] = [],
        pool: ThreadPoolExecutor = None,
) -> Generator:
    '''
    在线程池中批量运行任务，并将运行结果以生成器的形式返回。
    请确保任务中的所有操作是线程安全的，任务函数请全部使用关键字参数。
    '''
    tasks = []
    if pool is None:
        pool = ThreadPoolExecutor()

    for kwargs in params:
        thread = pool.submit(func, **kwargs)
        tasks.append(thread)

    for obj in as_completed(tasks):
        yield obj.result()


def files2docs_in_thread(
        files: List[Union[KnowledgeFile, Tuple[str, str], Dict]],
        pool: ThreadPoolExecutor = None,
) -> Generator:
    '''
    利用多线程批量将文件转化成langchain Document.
    生成器返回值为{(kb_name, file_name): docs}
    '''

    def task(*, file: KnowledgeFile, **kwargs) -> Dict[Tuple[str, str], List[Document]]:
        try:
            return True, (file.kb_name, file.filename, file.file2text(**kwargs))
        except Exception as e:
            return False, e

    kwargs_list = []
    for i, file in enumerate(files):
        kwargs = {}
        if isinstance(file, tuple) and len(file) >= 2:
            files[i] = KnowledgeFile(filename=file[0], knowledge_base_name=file[1])
        elif isinstance(file, dict):
            filename = file.pop("filename")
            kb_name = file.pop("kb_name")
            files[i] = KnowledgeFile(filename=filename, knowledge_base_name=kb_name)
            kwargs = file
        kwargs["file"] = file
        kwargs_list.append(kwargs)

    for result in run_in_thread_pool(func=task, params=kwargs_list, pool=pool):
        yield result
