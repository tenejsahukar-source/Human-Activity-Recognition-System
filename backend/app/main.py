from fastapi import FastAPI
from app.routes.predict import router as predict_router
from app.routes.camera import router as camera_router
from app.api.routes import router
from fastapi.middleware.cors import CORSMiddleware

from app.db.database import engine
from app.db.models import Base

import asyncio
import platform

# ✅ Windows fix
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print("✅ Set Windows event loop policy")

# ✅ CREATE APP FIRST
app = FastAPI()

# ✅ INCLUDE ROUTERS AFTER APP CREATION
app.include_router(predict_router)
app.include_router(camera_router)   # 🔥 CAMERA ADDED CORRECTLY
app.include_router(router)

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ DATABASE INIT
Base.metadata.create_all(bind=engine)