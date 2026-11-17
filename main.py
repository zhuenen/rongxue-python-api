
from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
from io import BytesIO

app = FastAPI(
    title="融学·智伴统一数据处理服务",
    version="2.0.0"
)

REQUIRED_COLUMNS = [
    "记录编号",
    "学生编号",
    "材料编号",
    "评分状态",
    "得分",
    "满分"
]


def read_csv_file(content):
    """兼容常见中文CSV编码"""
    for encoding in ["utf-8-sig", "gbk"]:
        try:
            return pd.read_csv(
                BytesIO(content),
                encoding=encoding
            )
        except UnicodeDecodeError:
            continue

    raise ValueError("无法识别CSV编码")


def clean_dataframe(df):
    """基础数据清洗"""

    df = df.copy()

    # 清理文本字段首尾空格
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )

    original_count = len(df)

    # 只删除完全相同的重复行
    df = df.drop_duplicates().copy()

    duplicate_count = original_count - len(df)

    # 检查记录编号重复，但不擅自删除不同版本记录
    duplicate_ids = []
    if "记录编号" in df.columns:
        duplicated = df[
            df["记录编号"].notna()
            & df["记录编号"].duplicated(keep=False)
        ]
        duplicate_ids = (
            duplicated["记录编号"].astype(str).unique().tolist()
        )

    # 数值字段转换
    df["得分"] = pd.to_numeric(
        df["得分"], errors="coerce"
    )
    df["满分"] = pd.to_numeric(
        df["满分"], errors="coerce"
    )

    # 严格区分评分状态
    scored_mask = df["评分状态"].eq("已评分")

    # 已评分记录中，检查分数是否有效
    invalid_mask = scored_mask & (
        df["得分"].isna()
        | df["满分"].isna()
        | (df["得分"] < 0)
        | (df["满分"] <= 0)
        | (df["得分"] > df["满分"])
    )

    invalid_records = df[invalid_mask].copy()

    # 只有已评分且分数有效的记录参与成绩统计
    valid_scored = df[
        scored_mask & ~invalid_mask
    ].copy()

    # 其他评分状态不按零分处理
    unscored = df[~scored_mask].copy()

    return {
        "cleaned": df,
        "valid_scored": valid_scored,
        "unscored": unscored,
        "invalid": invalid_records,
        "duplicate_count": duplicate_count,
        "duplicate_ids": duplicate_ids
    }


def summarize_by(df, column):
    """按学生、主题、节点或材料统计"""

    if column not in df.columns:
        return []

    data = df[
        df[column].notna()
        & (df[column].astype(str).str.strip() != "")
    ]

    results = []

    for name, group in data.groupby(column):

        total_score = float(group["得分"].sum())
        total_full = float(group["满分"].sum())

        score_rate = (
            round(total_score / total_full * 100, 2)
            if total_full > 0 else None
        )

        results.append({
            "编号": str(name),
            "有效评分记录数": int(len(group)),
            "总得分": round(total_score, 2),
            "总满分": round(total_full, 2),
            "得分率": score_rate,
            "满分记录数": int(
                (group["得分"] == group["满分"]).sum()
            ),
            "未满分记录数": int(
                (group["得分"] < group["满分"]).sum()
            )
        })

    return results


@app.get("/")
def home():
    return {
        "system": "融学·智伴04统一数据处理模块",
        "version": "2.0.0",
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

    if not (file.filename or "").lower().endswith(".csv"):
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
        df = read_csv_file(content)

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


@app.post("/upload/analyze")
async def upload_analyze(file: UploadFile = File(...)):

    if not (file.filename or "").lower().endswith(".csv"):
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
        df = read_csv_file(content)

        missing = [
            col for col in REQUIRED_COLUMNS
            if col not in df.columns
        ]

        if missing:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "缺少必要字段",
                    "missing_columns": missing
                }
            )

        result = clean_dataframe(df)

        cleaned = result["cleaned"]
        valid = result["valid_scored"]

        total_score = float(valid["得分"].sum())
        total_full = float(valid["满分"].sum())

        class_rate = (
            round(total_score / total_full * 100, 2)
            if total_full > 0 else None
        )

        return {
            "filename": file.filename,

            "data_quality": {
                "original_records": int(len(df)),
                "cleaned_records": int(len(cleaned)),
                "exact_duplicates_removed": int(
                    result["duplicate_count"]
                ),
                "duplicate_record_ids": result["duplicate_ids"],
                "invalid_scoring_records": int(
                    len(result["invalid"])
                ),
                "unscored_records": int(
                    len(result["unscored"])
                ),
                "valid_scored_records": int(len(valid))
            },

            "class_summary": {
                "total_score": round(total_score, 2),
                "total_full_score": round(total_full, 2),
                "class_score_rate": class_rate,
                "full_score_records": int(
                    (valid["得分"] == valid["满分"]).sum()
                ),
                "below_full_score_records": int(
                    (valid["得分"] < valid["满分"]).sum()
                )
            },

            "student_analysis": summarize_by(
                valid, "学生编号"
            ),

            "topic_analysis": summarize_by(
                valid, "主题编号"
            ),

            "node_analysis": summarize_by(
                valid, "节点编号"
            ),

            "material_analysis": summarize_by(
                valid, "材料编号"
            ),

            "validation": "分析完成"
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"数据处理失败：{str(e)}"
        )
