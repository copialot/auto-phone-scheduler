from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import inspect, text
from app.config import get_settings

settings = get_settings()

engine = create_async_engine(settings.database_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()


def _get_column_default_sql(column) -> str:
    """获取列的默认值 SQL"""
    if column.default is not None:
        default = column.default.arg
        if isinstance(default, bool):
            return "1" if default else "0"
        elif isinstance(default, str):
            return f"'{default}'"
        elif default is None:
            return "NULL"
        return str(default)
    return "NULL"


def _sync_add_missing_columns(conn):
    """同步添加缺失的列（在 run_sync 中调用）"""
    inspector = inspect(conn)

    for table_name, table in Base.metadata.tables.items():
        if not inspector.has_table(table_name):
            continue

        existing_columns = {col["name"] for col in inspector.get_columns(table_name)}

        for column in table.columns:
            if column.name not in existing_columns:
                col_type = column.type.compile(conn.dialect)
                nullable = "NULL" if column.nullable else "NOT NULL"
                default = _get_column_default_sql(column)

                # SQLite ALTER TABLE ADD COLUMN
                sql = f"ALTER TABLE {table_name} ADD COLUMN {column.name} {col_type} {nullable} DEFAULT {default}"
                conn.execute(text(sql))


async def init_db():
    async with engine.begin() as conn:
        # 先创建新表
        await conn.run_sync(Base.metadata.create_all)
        # 然后添加缺失的列
        await conn.run_sync(_sync_add_missing_columns)
