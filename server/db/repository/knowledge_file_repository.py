from server.db.models.knowledge_base_model import KnowledgeBaseModel
from server.db.models.knowledge_file_model import KnowledgeFileModel, FileDocModel
from server.db.session import with_session
from server.knowledge_base.utils import KnowledgeFile
from typing import List, Dict


@with_session
def list_docs_from_db(session,
                      kb_name: str,
                      file_name: str = None,
                      metadata: Dict = {},
                      ) -> List[Dict]:
    '''
    列出某知识库某文件对应的所有Document。
    返回形式：[{"id": str, "metadata": dict}, ...]
    '''
    docs = session.query(FileDocModel).filter_by(kb_name=kb_name)
    if file_name:
        docs = docs.filter_by(file_name=file_name)
    for k, v in metadata.items():
        docs = docs.filter(FileDocModel.meta_data[k].as_string()==str(v))

    return [{"id": x.doc_id, "metadata": x.metadata} for x in docs.all()]


@with_session
def delete_docs_from_db(session,
                      kb_name: str,
                      file_name: str = None,
                      ) -> List[Dict]:
    '''
    删除某知识库某文件对应的所有Document，并返回被删除的Document。
    返回形式：[{"id": str, "metadata": dict}, ...]
    '''
    docs = list_docs_from_db(kb_name=kb_name, file_name=file_name)
    query = session.query(FileDocModel).filter_by(kb_name=kb_name)
    if file_name:
        query = query.filter_by(file_name=file_name)
    query.delete()
    session.commit()
    return docs


@with_session
def add_docs_to_db(session,
                   kb_name: str,
                   file_name: str,
                   doc_infos: List[Dict]):
    '''
    将某知识库某文件对应的所有Document信息添加到数据库。
    doc_infos形式：[{"id": str, "metadata": dict}, ...]
    '''
    for d in doc_infos:
        obj = FileDocModel(
            kb_name=kb_name,
            file_name=file_name,
            doc_id=d["id"],
            meta_data=d["metadata"],
        )
        session.add(obj)
    return True


@with_session
def count_files_from_db(session, kb_name: str) -> int:
    return session.query(KnowledgeFileModel).filter_by(kb_name=kb_name).count()


@with_session
def list_files_from_db(session, kb_name):
    files = session.query(KnowledgeFileModel).filter_by(kb_name=kb_name).all()
    docs = [f.file_name for f in files]
    return docs


@with_session
def add_file_to_db(session,
                kb_file: KnowledgeFile,
                docs_count: int = 0, # 切割的数量 docs大小 17
                custom_docs: bool = False,
                doc_infos: List[str] = [], # 形式：[{"id": str, "metadata": dict}, ...]
                ):
    kb = session.query(KnowledgeBaseModel).filter_by(kb_name=kb_file.kb_name).first() # kb = <KnowledgeBase(id='2', kb_name='langchain', vs_type='milvus', embed_model='m3e-base', file_count='1', create_time='2023-09-05 10:18:59')>
    if kb:
        # 如果已经存在该文件，则更新文件信息与版本号  session是什么？是对话窗口中：知识库中包含源文件与向量库，请从下表中选择文件后操作  这个框里的数据
        existing_file: KnowledgeFileModel = (session.query(KnowledgeFileModel)
                                             .filter_by(file_name=kb_file.filename,
                                                        kb_name=kb_file.kb_name)
                                            .first())
        mtime = kb_file.get_mtime() # 获取文件上传时间
        size = kb_file.get_size() # 获取文件大小 单位B

        if existing_file:
            existing_file.file_mtime = mtime
            existing_file.file_size = size
            existing_file.docs_count = docs_count
            existing_file.custom_docs = custom_docs
            existing_file.file_version += 1
        # 否则，添加新文件 也是渲染·按session的
        else: # new_file = <KnowledgeFile(id='None', file_name='8月报.txt', file_ext='.txt', kb_name='langchain', document_loader_name='UnstructuredFileLoader', text_splitter_name='SpacyTextSplitter', file_version='None', create_time='None')>
            new_file = KnowledgeFileModel(
                file_name=kb_file.filename,
                file_ext=kb_file.ext,
                kb_name=kb_file.kb_name,
                document_loader_name=kb_file.document_loader_name,
                text_splitter_name=kb_file.text_splitter_name or "SpacyTextSplitter",
                file_mtime=mtime,
                file_size=size,
                docs_count = docs_count,
                custom_docs=custom_docs,
            )
            kb.file_count += 1
            session.add(new_file)
        add_docs_to_db(kb_name=kb_file.kb_name, file_name=kb_file.filename, doc_infos=doc_infos)
    return True


@with_session
def delete_file_from_db(session, kb_file: KnowledgeFile):
    existing_file = session.query(KnowledgeFileModel).filter_by(file_name=kb_file.filename,
                                                                kb_name=kb_file.kb_name).first()
    if existing_file:
        session.delete(existing_file)
        delete_docs_from_db(kb_name=kb_file.kb_name, file_name=kb_file.filename)
        session.commit()

        kb = session.query(KnowledgeBaseModel).filter_by(kb_name=kb_file.kb_name).first()
        if kb:
            kb.file_count -= 1
            session.commit()
    return True


@with_session
def delete_files_from_db(session, knowledge_base_name: str):
    session.query(KnowledgeFileModel).filter_by(kb_name=knowledge_base_name).delete()
    session.query(FileDocModel).filter_by(kb_name=knowledge_base_name).delete()
    kb = session.query(KnowledgeBaseModel).filter_by(kb_name=knowledge_base_name).first()
    if kb:
        kb.file_count = 0

    session.commit()
    return True


@with_session
def file_exists_in_db(session, kb_file: KnowledgeFile):
    existing_file = session.query(KnowledgeFileModel).filter_by(file_name=kb_file.filename,
                                                                kb_name=kb_file.kb_name).first()
    return True if existing_file else False


@with_session
def get_file_detail(session, kb_name: str, filename: str) -> dict:
    file: KnowledgeFileModel = (session.query(KnowledgeFileModel)
                                .filter_by(file_name=filename,
                                            kb_name=kb_name).first())
    if file:
        return {
            "kb_name": file.kb_name,
            "file_name": file.file_name,
            "file_ext": file.file_ext,
            "file_version": file.file_version,
            "document_loader": file.document_loader_name,
            "text_splitter": file.text_splitter_name,
            "create_time": file.create_time,
            "file_mtime": file.file_mtime,
            "file_size": file.file_size,
            "custom_docs": file.custom_docs,
            "docs_count": file.docs_count,
        }
    else:
        return {}
