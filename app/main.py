from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# import routes
from routes.main_routes import router as main_router
from routes.vis_routes import router as vis_router
from routes.data_routes import router as data_router

app = FastAPI()

# Mounting static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include the router
app.include_router(main_router)
app.include_router(vis_router, prefix="/vis")
app.include_router(data_router, prefix="/data")
