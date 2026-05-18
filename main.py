from datetime import datetime
from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
from pathlib import Path
from astrbot.api import AstrBotConfig, logger
from astrbot.api.message_components import Plain, Image
from astrbot.core.utils.astrbot_path import get_astrbot_data_path
import pyarrow as pa
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
                ]
            )
            self.db.create_table("memory", schema=schema)
            logger.warning("创建了 memory 表")
        else:
            logger.warning("memory 表已存在")
        
        self.table = self.db.open_table("memory")

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
        msg_chain = event.get_messages()
        for seg in msg_chain:
            if isinstance(seg, Plain):
                logger.warning(f"文本: {seg.text}")
                vector = await self.get_embedding(seg.text)
        
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
        "vector": vector
    }
]
        
        self.table.add(magical_characters)

    def bot_memory(self,event: AstrMessageEvent):
        pass