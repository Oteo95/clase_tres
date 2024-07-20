from uvicorn import run
import os
from fastapi import FastAPI, File, UploadFile
from typing import List
from pydantic import BaseModel

from fastapi.middleware.cors import CORSMiddleware


if __name__ == "__main__":
    run("server.api:app", host="0.0.0.0", port=8000, reload=True)