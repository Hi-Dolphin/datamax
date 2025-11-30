from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.router import get_api_router
from .core.config import get_settings
from .core.database import Base, get_engine, get_sessionmaker
from .core.logging import configure_logging
from .services.seeder import seed_demo_data


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings)

    app = FastAPI(title=settings.project_name)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api_router = get_api_router()
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.on_event("startup")
    async def on_startup() -> None:
        engine = get_engine(settings)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        session_factory = get_sessionmaker(settings)
        async with session_factory() as session:
            await seed_demo_data(session)
            await session.commit()

    @app.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
