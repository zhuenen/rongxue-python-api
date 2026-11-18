from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends, Query
from typing import Optional
from io import BytesIO
import os
import json
import uuid
import hmac
import logging
import pandas as pd
import numpy as np
import re
import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

app = FastAPI(
    title="融学·智伴统一数据处理服务",
    version="3.5.0",
    description="支持学生作答数据上传、清洗、版本处理、五表关联、知识指标及观察指标学情统计"
)

MAX_FILE_SIZE = 20 * 1024 * 1024

REQUIRED_COLUMNS = [
    "记录编号",
    "学生编号",
    "材料编号",
    "评分状态",
    "得分",
    "满分"
]

EXPECTED_COLUMNS = [
    "记录编号",
    "学生编号",
    "材料编号",
    "主题编号",
    "节点编号",
    "作答事件编号",
    "修订版本",
    "模拟提交时间",
    "评分状态",
    "得分",
    "满分",
    "学生最终作答",
    "解答过程",
    "评分点得分"
]

OPTIONAL_COLUMNS = [
    "主题编号",
    "节点编号",
    "作答事件编号",
    "修订版本",
    "模拟提交时间",
    "学生最终作答",
    "解答过程",
    "评分点得分"
]


# ==========================================
# 1. 基础数据读取
# ==========================================

def read_csv_file(content):
    """兼容常见中文CSV编码"""

    try:
        return pd.read_csv(
            BytesIO(content),
            encoding="utf-8-sig"
        )
    except UnicodeDecodeError:
        try:
            return pd.read_csv(
                BytesIO(content),
                encoding="gbk"
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"CSV读取失败：{str(e)}"
            )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"CSV读取失败：{str(e)}"
        )


def read_upload(content, filename):
    """检查文件类型和大小"""

    if not filename or not filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="请上传CSV文件"
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="文件不能超过20MB"
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="上传文件为空"
        )

    return read_csv_file(content)


async def load_optional_file(upload):
    """读取可选关联文件"""

    if upload is None:
        return None

    content = await upload.read()

    return read_upload(content, upload.filename)


def require_columns(df, columns, name="数据文件"):
    """必要字段检查"""

    missing = [
        col for col in columns
        if col not in df.columns
    ]

    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"{name}缺少必要字段",
                "missing_columns": missing
            }
        )


def clean_text_columns(df):
    """清理文本字段，不改变原始业务含义"""

    df = df.copy()

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )

        df[col] = df[col].replace("", np.nan)

    return df


def records_to_json(df):
    """转换成可以返回给API的JSON数据"""

    if df is None or df.empty:
        return []

    result = df.copy()

    for col in result.columns:
        if pd.api.types.is_datetime64_any_dtype(result[col]):
            series = result[col]

            if getattr(series.dt, "tz", None) is not None:
                series = series.dt.tz_convert("Asia/Shanghai")

            result[col] = series.apply(
                lambda x: x.isoformat() if pd.notna(x) else None
            )

    result = result.astype(object)
    result = result.where(pd.notna(result), None)

    return result.to_dict(orient="records")


# ==========================================
# 2. 统计工具
# ==========================================

def _group_key(value):
    """统一分组键，避免空值在全部事件与有效评分数据间匹配失败"""
    return "未分类" if pd.isna(value) else str(value)


def calculate_group_stats(df, group_column, valid_df=None):
    """
    按指定字段计算学情统计。

    df：该分组的全部正式作答事件，用于计算作答事件数。
    valid_df：全部有效评分记录，用于计算有效评分数、得分及得分率。
    """

    if df is None or df.empty or group_column not in df.columns:
        return []

    if valid_df is None:
        valid_df = df

    if valid_df is None or group_column not in valid_df.columns:
        valid_df = df.iloc[0:0].copy()

    valid_groups = {}

    if not valid_df.empty:
        for valid_name, valid_group in valid_df.groupby(
            group_column,
            dropna=False
        ):
            valid_groups[_group_key(valid_name)] = valid_group

    result = []

    for group_name, group in df.groupby(
        group_column,
        dropna=False
    ):
        key = _group_key(group_name)
        scored_group = valid_groups.get(key)

        if scored_group is None:
            scored_group = valid_df.iloc[0:0].copy()

        total_score = (
            pd.to_numeric(scored_group["得分"], errors="coerce").sum()
            if "得分" in scored_group.columns else 0
        )

        total_full = (
            pd.to_numeric(scored_group["满分"], errors="coerce").sum()
            if "满分" in scored_group.columns else 0
        )

        result.append({
            group_column: key,
            "作答事件数": int(len(group)),
            "有效评分数": int(len(scored_group)),
            "得分总和": round(float(total_score), 2),
            "满分总和": round(float(total_full), 2),
            "得分率": round(
                float(total_score / total_full * 100), 2
            ) if total_full > 0 else None
        })

    return result


def calculate_multi_group_stats(df, columns, valid_df=None):
    """按多个知识指标分别统计；事件数与评分指标使用不同数据口径"""

    result = {}

    for col in columns:
        if col in df.columns:
            result[col] = calculate_group_stats(
                df,
                col,
                valid_df=valid_df
            )

    return result


# ==========================================
# 3. 题目和知识指标映射
# ==========================================

def normalize_mapping(question_df: Optional[pd.DataFrame]):
    """标准化题目与材料映射表"""

    if question_df is None:
        return None, "未上传题目与材料表，跳过知识指标映射"

    require_columns(
        question_df,
        ["材料编号"],
        "题目与材料表"
    )

    keep = [
        c for c in [
            "材料编号",
            "主题编号",
            "指标编号",
            "知识点编号",
            "知识点名称",
            "题目或材料表",
            "题干或内容",
            "题型",
            "满分",
            "参考答案",
            "评分点"
        ]
        if c in question_df.columns
    ]

    q = question_df[keep].copy()

    q["材料编号"] = (
        q["材料编号"].astype(str).str.strip()
    )

    q = q.drop_duplicates(
        subset=["材料编号"],
        keep="last"
    )

    return q, "已按材料编号关联题目与知识指标"


def normalize_requirements(requirements_df):
    """标准化课程要求表，建立指标编号到课程目标/观察指标的映射"""

    if requirements_df is None:
        return None, "未上传课程要求表，未补充课程标准说明"

    require_columns(
        requirements_df,
        ["指标编号"],
        "课程要求表"
    )

    keep = [
        c for c in [
            "指标编号",
            "主题",
            "主题目标",
            "观察指标",
            "课标整理编号",
            "课标文件页／正文页",
            "教材正文页"
        ]
        if c in requirements_df.columns
    ]

    req = requirements_df[keep].copy()

    req["指标编号"] = (
        req["指标编号"].astype(str).str.strip()
    )

    req = req.dropna(
        subset=["指标编号"]
    ).drop_duplicates(
        subset=["指标编号"],
        keep="last"
    )

    return req, "已按指标编号关联课程要求与观察指标"


def explode_indicator_rows(df):
    """将一个材料对应多个观察指标的记录拆分，分别计入各指标统计"""

    if df is None or df.empty or "指标编号" not in df.columns:
        return df.copy() if df is not None else pd.DataFrame()

    result = df.copy()

    result["指标编号"] = (
        result["指标编号"]
        .fillna("")
        .astype(str)
        .str.split(r"[、,，;；]")
    )

    result = result.explode("指标编号")

    result["指标编号"] = (
        result["指标编号"].astype(str).str.strip()
    )

    result = result[
        result["指标编号"] != ""
    ].copy()

    return result


def merge_mapping(answer_df, mapping_df, key):
    """将映射信息关联到作答记录"""

    if mapping_df is None or key not in answer_df.columns:
        return answer_df

    if key not in mapping_df.columns:
        return answer_df

    mapping = mapping_df.copy()

    mapping[key] = (
        mapping[key].astype(str).str.strip()
    )

    mapping = mapping.drop_duplicates(
        subset=[key],
        keep="last"
    )

    extra_columns = [
        c for c in mapping.columns
        if c == key or c not in answer_df.columns
    ]

    return answer_df.merge(
        mapping[extra_columns],
        on=key,
        how="left"
    )


# ==========================================
# 4. 核心数据分析
# ==========================================

def analyze_payload(
    answer_df: pd.DataFrame,
    student_df=None,
    question_df=None,
    requirements_df=None,
    node_df=None
):

    require_columns(
        answer_df,
        REQUIRED_COLUMNS,
        "作答记录"
    )

    df = clean_text_columns(answer_df)

    original_count = len(df)
    original_columns = list(df.columns)

    exact_duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    optional_defaults = {
        "主题编号": None,
        "节点编号": None,
        "作答事件编号": None,
        "修订版本": 1,
        "模拟提交时间": None,
        "学生最终作答": None,
        "解答过程": None,
        "评分点得分": None
    }

    for col, default in optional_defaults.items():
        if col not in df.columns:
            df[col] = default

    df["得分"] = pd.to_numeric(
        df["得分"], errors="coerce"
    )

    df["满分"] = pd.to_numeric(
        df["满分"], errors="coerce"
    )

    df["修订版本"] = pd.to_numeric(
        df["修订版本"], errors="coerce"
    ).fillna(1)

    df["模拟提交时间"] = pd.to_datetime(
        df["模拟提交时间"],
        errors="coerce",
        utc=True
    )

    missing_event = (
        df["作答事件编号"].isna()
        | (
            df["作答事件编号"]
            .astype(str)
            .str.strip()
            .isin(["", "nan", "None"])
        )
    )

    df.loc[missing_event, "作答事件编号"] = (
        df.loc[missing_event, "学生编号"].astype(str)
        + "-"
        + df.loc[missing_event, "节点编号"].fillna(
            "未知"
        ).astype(str)
        + "-"
        + df.loc[missing_event, "材料编号"].astype(str)
    )

    df = df.sort_values(
        ["修订版本", "模拟提交时间"],
        na_position="first"
    )

    latest = df.drop_duplicates(
        subset=["作答事件编号"],
        keep="last"
    ).copy()

    history_count = len(df) - len(latest)

    latest["评分状态"] = (
        latest["评分状态"]
        .fillna("状态缺失")
        .astype(str)
        .str.strip()
    )

    status_counts = (
        latest["评分状态"].value_counts().to_dict()
    )

    # 未评分、待评分、未作答均不直接按零分处理
    latest["有效评分"] = (
        (latest["评分状态"] == "已评分")
        & latest["得分"].notna()
        & latest["满分"].notna()
        & (latest["满分"] > 0)
        & (latest["得分"] >= 0)
        & (latest["得分"] <= latest["满分"])
    )

    latest["异常原因"] = ""

    def add_reason(mask, reason):
        latest.loc[
            mask & (latest["异常原因"] == ""),
            "异常原因"
        ] = reason

    add_reason(
        latest["得分"].notna() & (latest["得分"] < 0),
        "得分为负数"
    )

    add_reason(
        latest["满分"].notna() & (latest["满分"] <= 0),
        "满分无效"
    )

    add_reason(
        latest["得分"].notna()
        & latest["满分"].notna()
        & (latest["得分"] > latest["满分"]),
        "得分超过满分"
    )

    add_reason(
        (latest["评分状态"] == "已评分")
        & (latest["得分"].isna() | latest["满分"].isna()),
        "已评分但得分或满分缺失"
    )

    valid = latest[
        latest["有效评分"]
    ].copy()

    total_score = valid["得分"].sum()
    total_full = valid["满分"].sum()

    overall_rate = (
        round(float(total_score / total_full * 100), 2)
        if total_full > 0
        else None
    )

    mapping, mapping_message = normalize_mapping(
        question_df
    )

    latest = merge_mapping(
        latest,
        mapping,
        "材料编号"
    )

    requirements, requirements_message = normalize_requirements(
        requirements_df
    )

    latest = merge_mapping(
        latest,
        requirements,
        "指标编号"
    )

    valid = latest[
        latest["有效评分"]
    ].copy()

    if node_df is not None:
        if "节点编号" in node_df.columns:
            node_df = clean_text_columns(node_df)

            latest = merge_mapping(
                latest,
                node_df,
                "节点编号"
            )

            valid = latest[
                latest["有效评分"]
            ].copy()

    class_stats = []
    student_info_status = "未上传学生信息表"

    if student_df is not None:
        student_df = clean_text_columns(student_df)

        if "学生编号" in student_df.columns:
            class_column = next(
                (
                    col for col in [
                        "模拟班级编号",
                        "班级编号",
                        "班级名称",
                        "班级"
                    ]
                    if col in student_df.columns
                ),
                None
            )

            student_columns = ["学生编号"]

            if class_column:
                student_columns.append(class_column)

            roster = student_df[
                student_columns
            ].drop_duplicates(
                subset=["学生编号"],
                keep="last"
            )

            latest = latest.merge(
                roster,
                on="学生编号",
                how="left"
            )

            valid = latest[
                latest["有效评分"]
            ].copy()

            student_info_status = "已关联学生信息"

            if class_column:
                class_stats = calculate_group_stats(
                    latest,
                    class_column,
                    valid_df=valid
                )

    student_stats = calculate_group_stats(
        latest,
        "学生编号",
        valid_df=valid
    )

    theme_stats = calculate_group_stats(
        latest,
        "主题编号",
        valid_df=valid
    )

    node_stats = calculate_group_stats(
        latest,
        "节点编号",
        valid_df=valid
    )

    material_stats = calculate_group_stats(
        latest,
        "材料编号",
        valid_df=valid
    )

    indicator_latest = explode_indicator_rows(latest)
    indicator_valid = explode_indicator_rows(valid)

    knowledge_stats = calculate_multi_group_stats(
        indicator_latest,
        [
            "指标编号",
            "主题",
            "观察指标",
            "知识点编号",
            "知识点名称"
        ],
        valid_df=indicator_valid
    )

    exceptions = latest[
        latest["异常原因"] != ""
    ].copy()

    invalid_scoring = latest[
        (latest["评分状态"] == "已评分")
        & (~latest["有效评分"])
    ].copy()

    ungraded = latest[
        latest["评分状态"] != "已评分"
    ].copy()

    unanswered_count = int(
        (latest["评分状态"] == "未作答").sum()
    )

    pending_count = int(
        (latest["评分状态"] == "待评分").sum()
    )

    other_ungraded_count = int(
        len(ungraded) - unanswered_count - pending_count
    )

    quality_report = {
        "原始记录数": int(original_count),
        "完全重复记录数": exact_duplicates,
        "去重后记录数": int(len(df)),
        "正式作答事件数": int(len(latest)),
        "历史修订版本数": int(history_count),
        "有效评分数": int(len(valid)),
        "无效评分数": int(len(invalid_scoring)),
        "未评分记录数": int(len(ungraded)),
        "未作答数": unanswered_count,
        "待评分数": pending_count,
        "其他未评分状态数": other_ungraded_count,
        "未评分状态分布": (
            ungraded["评分状态"].value_counts().to_dict()
            if not ungraded.empty else {}
        ),
        "异常记录数": int(len(exceptions)),
        "评分状态分布": status_counts,
        "原始字段": original_columns,
        "知识指标映射状态": mapping_message,
        "课程要求关联状态": requirements_message,
        "学生信息关联状态": student_info_status
    }

    return {
        "system": "融学·智伴04统一数据处理模块",
        "version": "3.5.0",
        "status": "analysis_completed",
        "summary": {
            "原始记录数": int(original_count),
            "正式作答事件数": int(len(latest)),
            "有效评分数": int(len(valid)),
            "未作答数": unanswered_count,
            "待评分数": pending_count,
            "异常评分数": int(len(invalid_scoring)),
            "有效得分总和": round(float(total_score), 2),
            "有效满分总和": round(float(total_full), 2),
            "整体得分率": overall_rate
        },
        "data_quality": quality_report,
        "student_stats": student_stats,
        "class_stats": class_stats,
        "theme_stats": theme_stats,
        "node_stats": node_stats,
        "material_stats": material_stats,
        "knowledge_stats": knowledge_stats,
        "exceptions": records_to_json(exceptions.head(100)),
        "invalid_scoring": records_to_json(invalid_scoring.head(100)),
        "ungraded_records": records_to_json(ungraded.head(100)),
        "latest_records_preview": records_to_json(latest.head(20)),
        "_database_records": records_to_json(latest)
    }
    # ==========================================
# 12. PostgreSQL数据库连接与初始化
# ==========================================

def get_database_url():
    """读取Render PostgreSQL连接地址"""

    value = os.getenv("DATABASE_URL")

    if not value:
        return None

    if value.startswith("postgres://"):
        value = value.replace(
            "postgres://",
            "postgresql://",
            1
        )

    return value


def get_db_connection():
    """创建PostgreSQL连接"""

    database_url = get_database_url()

    if not database_url:
        raise HTTPException(
            status_code=503,
            detail="未配置DATABASE_URL，数据库功能暂不可用"
        )

    try:
        return psycopg.connect(
            database_url,
            row_factory=dict_row,
            connect_timeout=10
        )
    except Exception as exc:
        logging.exception("PostgreSQL连接失败")
        raise HTTPException(
            status_code=503,
            detail="PostgreSQL连接失败，请检查Render数据库状态及DATABASE_URL"
        ) from exc


def jsonb(value):
    """将Python对象转换为PostgreSQL JSONB"""

    return Jsonb(
        value,
        dumps=lambda obj: json.dumps(
            obj,
            ensure_ascii=False,
            default=str
        )
    )


def init_database():
    """创建批次、标准化记录和原始文件表"""

    with get_db_connection() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_batches (
                batch_id UUID PRIMARY KEY,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                filename TEXT NOT NULL,
                system_version TEXT NOT NULL,
                summary JSONB NOT NULL DEFAULT '{}'::jsonb,
                data_quality JSONB NOT NULL DEFAULT '{}'::jsonb,
                result JSONB NOT NULL DEFAULT '{}'::jsonb
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_batch_records (
                id BIGSERIAL PRIMARY KEY,
                batch_id UUID NOT NULL
                    REFERENCES analysis_batches(batch_id)
                    ON DELETE CASCADE,
                record_no INTEGER NOT NULL,
                student_id TEXT,
                class_name TEXT,
                record_data JSONB NOT NULL,
                UNIQUE(batch_id, record_no)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS analysis_batch_files (
                id BIGSERIAL PRIMARY KEY,
                batch_id UUID NOT NULL
                    REFERENCES analysis_batches(batch_id)
                    ON DELETE CASCADE,
                file_role TEXT NOT NULL,
                filename TEXT NOT NULL,
                content BYTEA NOT NULL,
                size_bytes BIGINT NOT NULL,
                UNIQUE(batch_id, file_role)
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_batch_created_at
            ON analysis_batches(created_at DESC)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_batch_records_student
            ON analysis_batch_records(batch_id, student_id)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_batch_records_class
            ON analysis_batch_records(batch_id, class_name)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_batch_records_data
            ON analysis_batch_records USING GIN(record_data)
        """)


@app.on_event("startup")
def startup_database():
    """服务启动时初始化数据库"""

    if get_database_url():
        init_database()


# ==========================================
# 13. 数据查询接口认证
# ==========================================

def require_data_api_key(
    x_api_key: Optional[str] = Header(default=None)
):
    """验证数据查询接口密钥"""

    expected_key = os.getenv("DATA_API_KEY")

    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="尚未配置DATA_API_KEY"
        )

    if not x_api_key or not hmac.compare_digest(
        x_api_key,
        expected_key
    ):
        raise HTTPException(
            status_code=401,
            detail="API Key无效或缺失"
        )


# ==========================================
# 14. 系统首页与健康检查
# ==========================================

@app.get("/")
def home():

    return {
        "system": "融学·智伴04统一数据处理模块",
        "version": "3.5.0",
        "status": "running",
        "functions": [
            "CSV数据上传",
            "数据字段校验",
            "数据清洗",
            "完全重复记录去重",
            "作答修订版本处理",
            "有效评分识别",
            "学生学情统计",
            "班级学情统计",
            "主题学情统计",
            "节点学情统计",
            "材料学情统计",
            "知识指标映射",
            "课程要求与观察指标关联",
            "异常数据识别",
            "数据质量报告",
            "PostgreSQL持久化",
            "历史批次查询",
            "学生个人学情查询",
            "班级学情查询",
            "知识点学情查询"
        ]
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "message": "Python数据服务运行正常",
        "version": "3.5.0",
        "database_configured": bool(get_database_url())
    }


@app.get("/db/health")
def database_health():

    if not get_database_url():
        return {
            "database": "not_configured",
            "message": "未配置DATABASE_URL"
        }

    try:
        with get_db_connection() as conn:
            conn.execute("SELECT 1")

        return {
            "database": "connected",
            "message": "PostgreSQL连接正常"
        }

    except Exception:
        return {
            "database": "error",
            "message": "数据库连接失败"
        }


# ==========================================
# 15. CSV预览接口
# ==========================================

@app.post("/upload/preview")
async def upload_preview(
    file: UploadFile = File(...)
):

    content = await file.read()

    df = read_upload(
        content,
        file.filename
    )

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    return {
        "filename": file.filename,
        "total_records": int(len(df)),
        "columns": df.columns.tolist(),
        "missing_columns": missing,
        "validation": (
            "通过" if not missing else "未通过"
        ),
        "preview": records_to_json(
            df.head(5)
        )
    }


# ==========================================
# 16. 完整分析与持久化接口
# ==========================================

@app.post("/upload/analyze")
async def upload_analyze(
    file: UploadFile = File(...),
    student_file: Optional[UploadFile] = File(None),
    question_file: Optional[UploadFile] = File(None),
    requirements_file: Optional[UploadFile] = File(None),
    node_file: Optional[UploadFile] = File(None)
):

    answer_content = await file.read()

    answer_df = read_upload(
        answer_content,
        file.filename
    )

    raw_files = {
        "answers": (
            file.filename,
            answer_content
        )
    }

    optional_uploads = [
        (student_file, "students"),
        (question_file, "questions"),
        (requirements_file, "requirements"),
        (node_file, "nodes")
    ]

    optional_frames = {}

    for upload, role in optional_uploads:

        if upload is None:
            optional_frames[role] = None
            continue

        content = await upload.read()

        raw_files[role] = (
            upload.filename,
            content
        )

        optional_frames[role] = read_upload(
            content,
            upload.filename
        )

    result = analyze_payload(
        answer_df=answer_df,
        student_df=optional_frames["students"],
        question_df=optional_frames["questions"],
        requirements_df=optional_frames["requirements"],
        node_df=optional_frames["nodes"]
    )

    result["filename"] = file.filename

    database_records = result.pop(
        "_database_records",
        []
    )

    batch_info = save_analysis_batch(
        filename=file.filename,
        result=result,
        records=database_records,
        raw_files=raw_files
    )

    result["batch_id"] = batch_info["batch_id"]
    result["created_at"] = batch_info["created_at"]
    result["database_saved"] = True

    return result


# ==========================================
# 17. 数据库批次持久化
# ==========================================

def save_analysis_batch(
    filename,
    result,
    records,
    raw_files
):
    """保存分析结果、标准化记录和原始CSV"""

    batch_id = str(uuid.uuid4())

    system_version = result.get(
        "version",
        "3.5.0"
    )

    summary = result.get("summary", {})
    data_quality = result.get("data_quality", {})

    class_candidates = [
        "模拟班级编号",
        "班级编号",
        "班级名称",
        "班级"
    ]

    with get_db_connection() as conn:

        conn.execute(
            """
            INSERT INTO analysis_batches (
                batch_id,
                filename,
                system_version,
                summary,
                data_quality,
                result
            )
            VALUES (%s::uuid, %s, %s, %s, %s, %s)
            """,
            (
                batch_id,
                filename or "unknown.csv",
                system_version,
                jsonb(summary),
                jsonb(data_quality),
                jsonb(result)
            )
        )

        record_rows = []

        for index, record in enumerate(
            records or [],
            start=1
        ):

            student_id = record.get("学生编号")

            if student_id is not None:
                student_id = str(student_id)

            class_name = None

            for class_column in class_candidates:

                value = record.get(class_column)

                if value is not None and str(value).strip():
                    class_name = str(value)
                    break

            record_rows.append((
                batch_id,
                index,
                student_id,
                class_name,
                jsonb(record)
            ))

        if record_rows:

            with conn.cursor() as cur:

                cur.executemany(
                    """
                    INSERT INTO analysis_batch_records (
                        batch_id,
                        record_no,
                        student_id,
                        class_name,
                        record_data
                    )
                    VALUES (%s::uuid, %s, %s, %s, %s)
                    """,
                    record_rows
                )

        file_rows = []

        for role, file_info in raw_files.items():

            if not file_info:
                continue

            original_filename, content = file_info

            file_rows.append((
                batch_id,
                role,
                original_filename or f"{role}.csv",
                content,
                len(content)
            ))

        if file_rows:

            with conn.cursor() as cur:

                cur.executemany(
                    """
                    INSERT INTO analysis_batch_files (
                        batch_id,
                        file_role,
                        filename,
                        content,
                        size_bytes
                    )
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    file_rows
                )

        row = conn.execute(
            """
            SELECT created_at
            FROM analysis_batches
            WHERE batch_id = %s::uuid
            """,
            (batch_id,)
        ).fetchone()

    return {
        "batch_id": batch_id,
        "created_at": row["created_at"].isoformat()
    }
    # ==========================================
# 18. 通用批次查询辅助函数
# ==========================================

def validate_batch_id(batch_id):
    """验证批次UUID格式"""

    try:
        return str(uuid.UUID(batch_id))
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="batch_id格式不正确"
        )


def get_batch_result(batch_id):
    """读取指定批次分析结果"""

    parsed_id = validate_batch_id(batch_id)

    with get_db_connection() as conn:

        row = conn.execute(
            """
            SELECT
                batch_id::text AS batch_id,
                created_at,
                filename,
                system_version,
                summary,
                data_quality,
                result
            FROM analysis_batches
            WHERE batch_id = %s::uuid
            """,
            (parsed_id,)
        ).fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail="未找到该分析批次"
        )

    row["created_at"] = row["created_at"].isoformat()

    return row


# ==========================================
# 19. 历史批次列表、最新批次和批次详情
# ==========================================

@app.get("/batches")
def list_batches(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    _: None = Depends(require_data_api_key)
):

    with get_db_connection() as conn:

        total_row = conn.execute(
            """
            SELECT COUNT(*) AS total
            FROM analysis_batches
            """
        ).fetchone()

        rows = conn.execute(
            """
            SELECT
                batch_id::text AS batch_id,
                created_at,
                filename,
                system_version,
                summary,
                data_quality
            FROM analysis_batches
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
            """,
            (limit, offset)
        ).fetchall()

    return {
        "total": total_row["total"],
        "limit": limit,
        "offset": offset,
        "items": [
            {
                **row,
                "created_at": row["created_at"].isoformat()
            }
            for row in rows
        ]
    }


@app.get("/batches/latest")
def latest_batch(
    _: None = Depends(require_data_api_key)
):

    with get_db_connection() as conn:

        row = conn.execute(
            """
            SELECT
                batch_id::text AS batch_id,
                created_at,
                filename,
                system_version,
                summary,
                data_quality,
                result
            FROM analysis_batches
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail="数据库中还没有分析批次"
        )

    row["created_at"] = row["created_at"].isoformat()

    return row


@app.get("/batches/{batch_id}")
def batch_detail(
    batch_id: str,
    _: None = Depends(require_data_api_key)
):

    return get_batch_result(batch_id)


# ==========================================
# 20. 学生、班级、知识点及作答记录查询
# ==========================================

@app.get("/batches/{batch_id}/students/{student_id}")
def batch_student(
    batch_id: str,
    student_id: str,
    _: None = Depends(require_data_api_key)
):

    parsed_id = validate_batch_id(batch_id)

    batch = get_batch_result(parsed_id)

    with get_db_connection() as conn:

        rows = conn.execute(
            """
            SELECT record_data
            FROM analysis_batch_records
            WHERE batch_id = %s::uuid
              AND student_id = %s
            ORDER BY record_no
            """,
            (parsed_id, str(student_id))
        ).fetchall()

    stats = [
        item
        for item in batch["result"].get("student_stats", [])
        if str(item.get("学生编号")) == str(student_id)
    ]

    return {
        "batch_id": parsed_id,
        "student_id": student_id,
        "student_stats": stats,
        "records": [
            row["record_data"]
            for row in rows
        ],
        "record_count": len(rows)
    }


@app.get("/batches/{batch_id}/classes")
def batch_classes(
    batch_id: str,
    _: None = Depends(require_data_api_key)
):

    batch = get_batch_result(batch_id)

    return {
        "batch_id": batch["batch_id"],
        "class_stats": batch["result"].get(
            "class_stats",
            []
        )
    }


@app.get("/batches/{batch_id}/knowledge")
def batch_knowledge(
    batch_id: str,
    _: None = Depends(require_data_api_key)
):

    batch = get_batch_result(batch_id)

    return {
        "batch_id": batch["batch_id"],
        "knowledge_stats": batch["result"].get(
            "knowledge_stats",
            {}
        )
    }


@app.get("/batches/{batch_id}/records")
def batch_records(
    batch_id: str,
    student_id: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    _: None = Depends(require_data_api_key)
):

    parsed_id = validate_batch_id(batch_id)

    with get_db_connection() as conn:

        if student_id is not None:

            where = """
                batch_id = %s::uuid
                AND student_id = %s
            """

            params = (
                parsed_id,
                str(student_id)
            )

        else:

            where = "batch_id = %s::uuid"

            params = (parsed_id,)

        total = conn.execute(
            f"""
            SELECT COUNT(*) AS total
            FROM analysis_batch_records
            WHERE {where}
            """,
            params
        ).fetchone()["total"]

        rows = conn.execute(
            f"""
            SELECT
                record_no,
                student_id,
                class_name,
                record_data
            FROM analysis_batch_records
            WHERE {where}
            ORDER BY record_no
            LIMIT %s OFFSET %s
            """,
            params + (limit, offset)
        ).fetchall()

    return {
        "batch_id": parsed_id,
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": rows
    }


# ==========================================
# 21. 最新批次快捷查询
# ==========================================

@app.get("/students/{student_id}/latest")
def latest_student(
    student_id: str,
    _: None = Depends(require_data_api_key)
):

    with get_db_connection() as conn:

        batch = conn.execute(
            """
            SELECT
                batch_id::text AS batch_id,
                created_at,
                result
            FROM analysis_batches
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()

        if not batch:
            raise HTTPException(
                status_code=404,
                detail="数据库中还没有分析批次"
            )

        rows = conn.execute(
            """
            SELECT record_data
            FROM analysis_batch_records
            WHERE batch_id = %s::uuid
              AND student_id = %s
            ORDER BY record_no
            """,
            (
                batch["batch_id"],
                str(student_id)
            )
        ).fetchall()

    stats = [
        item
        for item in batch["result"].get("student_stats", [])
        if str(item.get("学生编号")) == str(student_id)
    ]

    return {
        "batch_id": batch["batch_id"],
        "created_at": batch["created_at"].isoformat(),
        "student_id": student_id,
        "student_stats": stats,
        "records": [
            row["record_data"]
            for row in rows
        ]
    }


@app.get("/classes/latest")
def latest_classes(
    _: None = Depends(require_data_api_key)
):

    with get_db_connection() as conn:

        row = conn.execute(
            """
            SELECT
                batch_id::text AS batch_id,
                result
            FROM analysis_batches
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail="数据库中还没有分析批次"
        )

    return {
        "batch_id": row["batch_id"],
        "class_stats": row["result"].get(
            "class_stats",
            []
        )
    }


@app.get("/knowledge/latest")
def latest_knowledge(
    _: None = Depends(require_data_api_key)
):

    with get_db_connection() as conn:

        row = conn.execute(
            """
            SELECT
                batch_id::text AS batch_id,
                result
            FROM analysis_batches
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail="数据库中还没有分析批次"
        )

    return {
        "batch_id": row["batch_id"],
        "knowledge_stats": row["result"].get(
            "knowledge_stats",
            {}
        )
    }


# ==========================================
# 22. 数据库表检查接口
# ==========================================

@app.get("/database/tables")
def database_tables(
    _: None = Depends(require_data_api_key)
):

    with get_db_connection() as conn:

        rows = conn.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name IN (
                  'analysis_batches',
                  'analysis_batch_records',
                  'analysis_batch_files'
              )
            ORDER BY table_name
            """
        ).fetchall()

    return {
        "tables": [
            row["table_name"]
            for row in rows
        ],
        "count": len(rows)
    }
