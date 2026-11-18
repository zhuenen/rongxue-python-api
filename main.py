from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Optional
from io import BytesIO
import pandas as pd
import numpy as np

app = FastAPI(
    title="融学·智伴统一数据处理服务",
    version="3.1.0",
    description="支持学生作答数据上传、清洗、版本处理、知识点映射和学情统计"
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
            result[col] = result[col].apply(
                lambda x: x.isoformat() if pd.notna(x) else None
            )

    result = result.astype(object)
    result = result.where(pd.notna(result), None)

    return result.to_dict(orient="records")


# ==========================================
# 2. 统计工具
# ==========================================

def calculate_group_stats(df, group_column):
    """按照指定字段计算学情统计"""

    if df.empty or group_column not in df.columns:
        return []

    result = []

    for group_name, group in df.groupby(
        group_column,
        dropna=False
    ):

        total_score = group["得分"].sum()
        total_full = group["满分"].sum()

        result.append({
            group_column: (
                str(group_name)
                if pd.notna(group_name)
                else "未分类"
            ),
            "作答事件数": int(len(group)),
            "有效评分数": int(len(group)),
            "得分总和": round(float(total_score), 2),
            "满分总和": round(float(total_full), 2),
            "得分率": round(
                float(total_score / total_full * 100), 2
            ) if total_full > 0 else None
        })

    return result


def calculate_multi_group_stats(df, columns):
    """按照多个知识指标分别统计"""

    result = {}

    for col in columns:
        if col in df.columns:
            result[col] = calculate_group_stats(df, col)

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


def merge_mapping(answer_df, mapping_df, key):
    """将映射信息关联到作答记录"""

    if mapping_df is None or key not in answer_df.columns:
        return answer_df

    if key not in mapping_df.columns:
        return answer_df

    mapping = mapping_df.copy()

    mapping[key] = mapping[key].astype(str).str.strip()

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

    # 4.1 文本清洗
    df = clean_text_columns(answer_df)

    original_count = len(df)
    original_columns = list(df.columns)

    # 4.2 完全重复记录处理
    exact_duplicates = int(df.duplicated().sum())

    df = df.drop_duplicates().copy()

    # 4.3 补充可选字段
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

    # 4.4 数值转换
    df["得分"] = pd.to_numeric(
        df["得分"],
        errors="coerce"
    )

    df["满分"] = pd.to_numeric(
        df["满分"],
        errors="coerce"
    )

    df["修订版本"] = pd.to_numeric(
        df["修订版本"],
        errors="coerce"
    ).fillna(1)

    # 4.5 时间转换
    df["模拟提交时间"] = pd.to_datetime(
        df["模拟提交时间"],
        errors="coerce",
        utc=True
    )

    # 4.6 生成作答事件编号
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

    # 4.7 修订版本处理
    df = df.sort_values(
        ["修订版本", "模拟提交时间"],
        na_position="first"
    )

    latest = df.drop_duplicates(
        subset=["作答事件编号"],
        keep="last"
    ).copy()

    history_count = len(df) - len(latest)

    # 4.8 评分状态统计
    latest["评分状态"] = (
        latest["评分状态"]
        .fillna("状态缺失")
        .astype(str)
        .str.strip()
    )

    status_counts = (
        latest["评分状态"]
        .value_counts()
        .to_dict()
    )

    # 4.9 有效评分识别
    # 未评分、待评分、未作答均不直接按零分处理

    latest["有效评分"] = (
        (latest["评分状态"] == "已评分")
        & latest["得分"].notna()
        & latest["满分"].notna()
        & (latest["满分"] > 0)
        & (latest["得分"] >= 0)
        & (latest["得分"] <= latest["满分"])
    )

    # 4.10 异常原因识别
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

    # 4.11 有效评分数据
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

    # ======================================
    # 5. 关联题目和知识指标
    # ======================================

    mapping, mapping_message = normalize_mapping(
        question_df
    )

    latest = merge_mapping(
        latest,
        mapping,
        "材料编号"
    )

    valid = latest[
        latest["有效评分"]
    ].copy()

    # ======================================
    # 6. 关联节点信息
    # ======================================

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

    # ======================================
    # 7. 关联学生信息
    # ======================================

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
                    valid,
                    class_column
                )

    # ======================================
    # 8. 学生、主题、节点、材料统计
    # ======================================

    student_stats = calculate_group_stats(
        valid,
        "学生编号"
    )

    theme_stats = calculate_group_stats(
        valid,
        "主题编号"
    )

    node_stats = calculate_group_stats(
        valid,
        "节点编号"
    )

    material_stats = calculate_group_stats(
        valid,
        "材料编号"
    )

    # 知识指标统计
    knowledge_stats = calculate_multi_group_stats(
        valid,
        [
            "指标编号",
            "知识点编号",
            "知识点名称"
        ]
    )

    # ======================================
    # 9. 异常和无效评分记录
    # ======================================

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

    # ======================================
    # 10. 数据质量报告
    # ======================================

    quality_report = {
        "原始记录数": int(original_count),
        "完全重复记录数": exact_duplicates,
        "去重后记录数": int(len(df)),
        "正式作答事件数": int(len(latest)),
        "历史修订版本数": int(history_count),
        "有效评分数": int(len(valid)),
        "无效评分数": int(len(invalid_scoring)),
        "未评分记录数": int(len(ungraded)),
        "未评分状态分布": (
            ungraded["评分状态"].value_counts().to_dict()
            if not ungraded.empty else {}
        ),
        "异常记录数": int(len(exceptions)),
        "评分状态分布": status_counts,
        "原始字段": original_columns,
        "知识指标映射状态": mapping_message,
        "学生信息关联状态": student_info_status
    }

    # ======================================
    # 11. 输出标准化结果
    # ======================================

    return {
        "system": "融学·智伴04统一数据处理模块",
        "version": "3.1.0",
        "status": "analysis_completed",

        "summary": {
            "原始记录数": int(original_count),
            "正式作答事件数": int(len(latest)),
            "有效评分数": int(len(valid)),
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

        "exceptions": records_to_json(
            exceptions.head(100)
        ),

        "invalid_scoring": records_to_json(
            invalid_scoring.head(100)
        ),

        "ungraded_records": records_to_json(
            ungraded.head(100)
        ),

        "latest_records_preview": records_to_json(
            latest.head(20)
        )
    }


# ==========================================
# 12. API接口：系统首页
# ==========================================

@app.get("/")
def home():

    return {
        "system": "融学·智伴04统一数据处理模块",
        "version": "3.1.0",
        "status": "running",
        "functions": [
            "CSV数据上传",
            "数据字段校验",
            "数据清洗",
            "完全重复记录去重",
            "作答修订版本处理",
            "有效评分识别（仅统计已评分且分值有效的记录；待评分、未评分、未作答不计为零分）",
            "学生学情统计",
            "班级学情统计",
            "主题学情统计",
            "节点学情统计",
            "材料学情统计",
            "知识指标映射",
            "异常数据识别",
            "数据质量报告"
        ]
    }


# ==========================================
# 13. API接口：健康检查
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "message": "Python数据服务运行正常",
        "version": "3.1.0"
    }


# ==========================================
# 14. API接口：上传和预览
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
# 15. API接口：完整数据分析
# ==========================================

@app.post("/upload/analyze")
async def upload_analyze(

    file: UploadFile = File(...),

    student_file: Optional[UploadFile] = File(None),

    question_file: Optional[UploadFile] = File(None),

    requirements_file: Optional[UploadFile] = File(None),

    node_file: Optional[UploadFile] = File(None)

):

    # 读取作答记录
    answer_content = await file.read()

    answer_df = read_upload(
        answer_content,
        file.filename
    )

    # 读取可选关联文件
    student_df = await load_optional_file(
        student_file
    )

    question_df = await load_optional_file(
        question_file
    )

    requirements_df = await load_optional_file(
        requirements_file
    )

    node_df = await load_optional_file(
        node_file
    )

    # 执行分析
    result = analyze_payload(
        answer_df=answer_df,
        student_df=student_df,
        question_df=question_df,
        requirements_df=requirements_df,
        node_df=node_df
    )

    result["filename"] = file.filename

    return result
