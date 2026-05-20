from datetime import datetime
from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from astrbot.api.provider import ProviderRequest
from pathlib import Path
from astrbot.api import AstrBotConfig, logger
from astrbot.api.message_components import Plain, Image, At
from astrbot.core.utils.astrbot_path import get_astrbot_data_path
from astrbot.core.agent.message import TextPart 
import pyarrow as pa
import pandas as pd
import lancedb
import openai



class MyPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.name = "long_term_memory"
        
                
        raw_api_key = self.config.get("api_key")
        if isinstance(raw_api_key, (tuple, list)):
            raw_api_key = raw_api_key[0] if len(raw_api_key) > 0 else ""
        self.api_key = str(raw_api_key) if raw_api_key else ""

        raw_base_url = self.config.get("api_base_url")
        if isinstance(raw_base_url, (tuple, list)):
            raw_base_url = raw_base_url[0] if len(raw_base_url) > 0 else ""
        self.base_url = str(raw_base_url) if raw_base_url else ""
        
        self.embedding_model_name = self.config.get("embedding_model")
        self.dim = self.config.get("embedding_dim")
        self._openai_client = None

        # 构建数据路径
        plugin_data_path = Path(get_astrbot_data_path()) / "plugin_data" / self.name
        uri = str(plugin_data_path)
        self.db = lancedb.connect(uri)
        
        # 建表
        table_names = self.db.table_names()
        if "memory" not in table_names:
            schema = pa.schema(
                [
                    pa.field("time", pa.string()),
                    pa.field("group", pa.string()),
                    pa.field("sender", pa.string()),
                    pa.field("id", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), self.dim)),
                    pa.field("message", pa.string()),
                ]
            )
            self.db.create_table("memory", schema=schema)
            logger.warning("创建了 memory 表")
        else:
            logger.warning("memory 表已存在")
        
        self.table = self.db.open_table("memory")

            # ========== 原消息存储与向量查询 ==========

    def _ensure_client(self):
        """懒加载：第一次用到时才创建 client"""
        if self._openai_client is not None:
            return
        
        if not self.api_key:
            logger.warning("embedding_api_key 为空，请在 metadata.yaml 或 WebUI 配置中填写")
            return
        
        self._openai_client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url if self.base_url else None,
        )
        logger.warning("OpenAI client 已创建")

    async def get_embedding(self, text: str):
        self._ensure_client()
        if not self._openai_client:
            return None
        
        resp = await self._openai_client.embeddings.create(
            model=self.embedding_model_name,
            input=text
        )
        return resp.data[0].embedding

    def simple_time(self,ts) -> str:
        dt = datetime.fromtimestamp(ts)
        return f"{dt.year}/{dt.month:02d}/{dt.day:02d} {dt.hour:02d}:{dt.minute:02d}"
    
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def get_message(self, event: AstrMessageEvent):
        vector = None
        msg_chain = event.get_messages()
        for seg in msg_chain:
            if isinstance(seg, Plain):
                logger.warning(f"文本: {seg.text}")
                vector = await self.get_embedding(seg.text)

        if vector is None:
            logger.warning("没有文本内容或 embedding 失败，跳过存储")
            return
        
        time = self.simple_time(event.created_at)
        group = event.session_id
        sender = event.get_sender_name()
        id = event.get_sender_id()

        magical_characters = [
            {
                "time": time,
                "group": group,
                "sender": sender,
                "id": id,
                "vector": vector,
                "message": seg.text
            }
        ]
            
        self.table.add(magical_characters)
        logger.warning(f"已存储消息向量,{time},群: {group}, 发送者: {sender}/{id}")

    @filter.on_llm_request()
    async def reply_memory(self,event: AstrMessageEvent,req: ProviderRequest):
        current_message = event.get_messages()
        current_message_vector = None
        texts = []
        for seg in current_message:
            if isinstance(seg, Plain):
                texts.append(seg.text)
            elif isinstance(seg, At):
                name = seg.name or str(seg.qq)
                texts.append(f"[@{name}]")
            # 其他类型跳过，不要 return
        
        full_text = "".join(texts).strip()
        if not full_text:
            logger.warning("没有可 embedding 的文本，跳过记忆召回")
            return
        
        logger.warning(f"当前消息文本: {full_text}")
        current_message_vector = await self.get_embedding(full_text)
        if current_message_vector is None:
            return
        r2 = (
            self.table.search(current_message_vector)
            .select(["time", "group", "sender", "id","message"])
            .limit(8)
            .to_pandas()
        )
        r2 = r2.iloc[1:]                 # ← 从第2行开始取
        #r2 = r2[r2["_distance"] < 0.75].head(5)
        r2 = self._filter_duplicate_messages(r2) 
        r2 = r2[r2["message"].astype(str).str.len() > 3]
        logger.warning(r2) 

        
        if r2.empty:
            return "[记忆查询] 未找到相关记忆"


        current_group = event.session_id
        memory_texts = []
        for _, row in r2.iterrows():
            group_str = str(row["group"])
            # 如果是当前群，加上标记
            if group_str == current_group:
                group_str += "<-这是本群"
            
            memory_texts.append(
                f"[{row['time']}] 群：{group_str} {row['sender']}({row['id']}): {row['message']}"
            )
        memory_block = "\n".join(memory_texts)
        logger.warning(memory_block) 
        
        prefix = f"【以下是你回忆起的对话片段，按照时间越近与相关性越高从上往下排序，你可以使用他们辅助进行回复，不过可能很多垃圾信息，可以随时使用记忆功能的工具来搜索记忆】\n{memory_block}\n\n"
        req.extra_user_content_parts.append(
            TextPart(text=prefix).mark_as_temp()
        )
        logger.warning(req)

    def _filter_duplicate_messages(self, df):
        """过滤内容完全重复的消息，保留每组的第一条和最后一条，保持原始召回顺序。"""
        if df.empty or "message" not in df.columns:
            return df
        
        # 标记所有重复内容（包括第一次出现）
        dup_mask = df["message"].duplicated(keep=False)
        unique_df = df[~dup_mask]          # 不重复的全保留
        dup_df = df[dup_mask]              # 重复的单独处理
        
        if dup_df.empty:
            return df
        
        def keep_first_last(group):
            # 组内 ≤2 条全保留；>2 条只保留首尾
            return group if len(group) <= 2 else group.iloc[[0, -1]]
        
        # group_keys=False 避免多级索引；sort=False 保持 LanceDB 召回的原始相似度顺序
        dup_filtered = dup_df.groupby("message", group_keys=False, sort=False).apply(keep_first_last)
        
        # 按原始索引排序，恢复顺序
        return pd.concat([unique_df, dup_filtered]).sort_index()

            # ==========    关系图谱    ==========
            

            
            # ========== 人格画像与好感度 ==========

    
    
            # ========== AI 自主调用工具 ==========

    @filter.llm_tool(name="query_memory")
    async def query_memory(self, event: AstrMessageEvent, query: str, limit: int = 5) -> str:
        '''查询长期记忆数据库，回忆过去群聊或用户相关的历史信息。
        Args:
            query(string): 查询意图描述或关键词，用于语义相似度匹配。
            limit(number): 返回条数上限，默认5条，最多10条。
        '''
        vector = await self.get_embedding(query)
        if not vector:
            return "[记忆查询失败] embedding 服务不可用"

        try:
            # 多取一条防意外，但返回时按用户要的 limit
            df = (
                self.table.search(vector)
                .select(["time", "group", "sender", "id", "message"])
                .limit(min(limit, 10) + 1)
                .to_pandas()
            )
            if df.empty:
                return "[记忆查询] 未找到相关记忆"

            texts = []
            for _, row in df.head(limit).iterrows():
                group_str = str(row["group"])
                if group_str == event.session_id:
                    group_str += "<-当前群"
                texts.append(
                    f"[{row['time']}] 群:{group_str} {row['sender']}({row['id']}): {row['message']}"
                )
            return "【查询到的历史记忆】\n" + "\n".join(texts)
        except Exception as e:
            logger.error(f"query_memory tool 出错: {e}")
            return f"[记忆查询错误] {e}"

    @filter.llm_tool(name="write_memory")
    async def write_memory(self, event: AstrMessageEvent, content: str) -> str:
        '''将重要信息主动写入长期记忆数据库。
        当用户明确表达偏好、身份、重要约定、关键事实，或你意识到"这件事以后可能有用"时使用。
        Args:
            content(string): 需要记忆的文本内容，建议简洁准确，去除口语化冗余。
        '''
        vector = await self.get_embedding(content)
        if not vector:
            return "[记忆写入失败] embedding 服务不可用"

        try:
            time_str = self.simple_time(datetime.now().timestamp())
            record = {
                "time": time_str,
                "group": event.session_id,
                "sender": "你自己",
                "id": str(event.get_sender_id()),
                "vector": vector,
                "message": content,
            }
            self.table.add([record])
            logger.warning(f"AI_TOOL 写入记忆: {content[:60]}")
            return f"[记忆已写入] {time_str} | {content[:100]}"
        except Exception as e:
            logger.error(f"write_memory tool 出错: {e}")
            return f"[记忆写入错误] {e}"
