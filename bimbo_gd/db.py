"""SQL connection through SQLAlchemy. Supports SQL Server, PostgreSQL, MySQL and SQLite."""
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Engine

from . import config


def get_engine() -> Engine:
    d = config.DB_DIALECT

    if d == "sqlite":
        return create_engine(f"sqlite:///{config.DB_SQLITE_PATH}")

    port = int(config.DB_PORT) if config.DB_PORT else None

    if d == "mssql":
        query = {"driver": config.DB_ODBC_DRIVER, "TrustServerCertificate": "yes"}
        if config.DB_TRUSTED:
            query["trusted_connection"] = "yes"
            user = password = None
        else:
            user, password = config.DB_USER, config.DB_PASSWORD
        url = URL.create(
            "mssql+pyodbc", username=user, password=password, host=config.DB_HOST,
            port=port, database=config.DB_NAME, query=query,
        )
        return create_engine(url)

    drivers = {"postgresql": "postgresql+psycopg2", "mysql": "mysql+pymysql"}
    if d not in drivers:
        raise ValueError(f"Unsupported DB_DIALECT: {d}")
    url = URL.create(
        drivers[d], username=config.DB_USER, password=config.DB_PASSWORD,
        host=config.DB_HOST, port=port, database=config.DB_NAME,
    )
    return create_engine(url)


def ping(engine: Engine) -> int:
    with engine.connect() as conn:
        return conn.execute(text("SELECT 1")).scalar()
