import fitz
import re
import math
import time
import base64
import os
import io
import uuid as _uuid
import json
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image as PILImage
import pdf2image
import cv2
import easyocr
import google.generativeai as genai

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[INIT] python-dotenv not installed — skipping .env load")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Database & Storage ────────────────────────────────────────────────────────
supabase_client = None
try:
    from supabase import create_client as _sb_create
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if url and key:
        supabase_client = _sb_create(url, key)
        print("[INIT] Supabase ready")
except Exception as e:
    print(f"[INIT] Supabase error: {e}")

r2_client = None
try:
    import boto3 as _boto3
    from botocore.config import Config as _boto_config
    
    acc_id = os.getenv("R2_ACCOUNT_ID")
    if acc_id:
        r2_client = _boto3.client(
            "s3",
            endpoint_url=f"https://{acc_id}.r2.cloudflarestorage.com",
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
            config=_boto_config(signature_version="s3v4"),
            region_name="auto"
        )
        print("[INIT] Cloudflare R2 ready")
except Exception as e:
    print(f"[INIT] R2 error: {e}")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Gemini Config ─────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("[INIT] Gemini API configured")

# ── OCR Reader Cache ──────────────────────────────────────────────────────────
print("[STARTUP] Loading EasyOCR model...")
try:
    _OCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
    print("[STARTUP] EasyOCR ready")
except Exception as e:
    print(f"[STARTUP] EasyOCR failed to load: {e}")
    _OCR_READER = None

def get_ocr_reader():
    return _OCR_READER

# ── Constants ─────────────────────────────────────────────────────────────────
MEMBER_COLORS = {
    "beam":   "#EC4899",
    "column": "#3B82F6",
    "brace":  "#F59E0B",
}

STEEL_PATTERNS = [
    r'W\d+[Xx]\d+',
    r'HSS\d+[Xx]\d+',
    r'L\d+[Xx]\d+',
    r'C\d+[Xx]\d+',
    r'MC\d+[Xx]\d+',
    r'ISA[\dXx]+',
]

ZOOM_FACTORS = [1.0, 2.0, 3.0]
COLUMN_PATTERNS = [r'W14X\d+', r'W12X\d+', r'W10X\d+', r'W8X\d+', r'HSS\d+[Xx]\d+']
ASSOCIATION_RADIUS_PIXELS = 120

# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_profile(text: str) -> str:
    t = text.upper().strip()
    t = re.sub(r'[^A-Z0-9X/]', '', t)
    t = re.sub(r'^[Vv][Vv]', "W", t)
    t = re.sub(r'^[Vv](?=\d)', "W", t)
    return t

def classify_from_profile(profile: str) -> str:
    p = profile.upper()
    if any(re.match(cp, p) for cp in COLUMN_PATTERNS):
        return "column"
    if re.match(r'ISA', p):
        return "brace"
    return "beam"

def preprocess_for_ocr(img):
