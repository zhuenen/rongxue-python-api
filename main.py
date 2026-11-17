
from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
from io import BytesIO

app = FastAPI(
    title="融学·智伴统一数据处理服务",
    version="1.0.0"
)

REQUIRED_COLUMNS = [
    "记录编号",
    "学生编号",
    "材料编号",
    "评分状态",
    "得分",
    "满分"
]


@app.get("/")
def home():
    return {
        "system": "融学·智伴04统一数据处理模块",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "message": "Python数据服务运行正常"
    }


@app.post("/upload/preview")
async def upload_preview(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="请上传CSV文件"
        )

    content = await file.read()

    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="文件不能超过20MB"
        )

    try:
        try:
            df = pd.read_csv(
                BytesIO(content),
                encoding="utf-8-sig"
            )
        except UnicodeDecodeError:
            df = pd.read_csv(
                BytesIO(content),
                encoding="gbk"
            )

        missing = [
            col for col in REQUIRED_COLUMNS
            if col not in df.columns
        ]

        return {
            "filename": file.filename,
            "total_records": len(df),
            "columns": df.columns.tolist(),
            "missing_columns": missing,
            "validation": "通过" if not missing else "未通过",
            "preview": df.head(5).fillna("").to_dict(
                orient="records"
            )
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"CSV读取失败：{str(e)}"
        )
