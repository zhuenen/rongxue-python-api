
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Optional
import pandas as pd
from io import BytesIO

app = FastAPI(
    title="融学·智伴统一数据处理服务",
    version="2.0.0",
    description="支持学生作答数据上传、清洗、版本处理和学情统计"
)

# 必须具备的核心字段
REQUIRED_COLUMNS = [
    "记录编号",
    "学生编号",
    "材料编号",
    "评分状态",
    "得分",
    "满分"
]

# 当前比赛数据的完整字段
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


def read_upload(file_content, filename):
    """读取上传的CSV文件"""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="请上传CSV文件"
        )

    if len(file_content) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="文件不能超过20MB"
        )

    return read_csv_file(file_content)


def records_to_json(df):
    """将DataFrame转换为可序列化数据"""
    if df.empty:
        return []

    result = df.copy()
    result = result.astype(object)
    result = result.where(pd.notna(result), None)
    return result.to_dict(orient="records")


def calculate_group_stats(df, group_column):
    """按指定字段统计学情"""
    if group_column not in df.columns:
        return []

    if df.empty:
        return []

    result = []

    for group_name, group in df.groupby(
        group_column,
        dropna=False
    ):
        total_score = group["得分"].sum()
        total_full = group["满分"].sum()

        result.append({
            group_column: str(group_name),
            "作答事件数": int(len(group)),
            "有效评分数": int(len(group)),
            "得分总和": round(float(total_score), 2),
            "满分总和": round(float(total_full), 2),
            "得分率": round(
                float(total_score / total_full * 100), 2
            ) if total_full > 0 else None
        })

    return result


@app.get("/")
def home():
    return {
        "system": "融学·智伴04统一数据处理模块",
        "version": "2.0.0",
        "status": "running",
        "functions": [
            "CSV上传预览",
            "数据清洗",
            "修订版本处理",
            "学生学情统计",
            "主题学情统计",
            "节点学情统计",
            "材料学情统计"
        ]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "message": "Python数据服务运行正常",
        "version": "2.0.0"
    }


@app.post("/upload/preview")
async def upload_preview(file: UploadFile = File(...)):
    """保留原有上传预览功能"""

    content = await file.read()
    df = read_upload(content, file.filename)

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    return {
        "filename": file.filename,
        "total_records": int(len(df)),
        "columns": df.columns.tolist(),
        "missing_columns": missing,
        "validation": "通过" if not missing else "未通过",
        "preview": records_to_json(df.head(5))
    }


@app.post("/upload/analyze")
async def upload_analyze(
    file: UploadFile = File(...),
    student_file: Optional[UploadFile] = File(None)
):
    """上传作答记录并完成清洗与学情统计"""

    content = await file.read()
    df = read_upload(content, file.filename)

    # 1. 字段检查
    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "缺少必要字段，无法分析",
                "missing_columns": missing
            }
        )

    original_count = len(df)

    # 2. 完全重复记录检查
    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    # 3. 补充可选字段，兼容不同批次数据
    optional_columns = [
        "主题编号",
        "节点编号",
        "作答事件编号",
        "修订版本",
        "模拟提交时间",
        "学生最终作答",
        "解答过程",
        "评分点得分"
    ]

    for col in optional_columns:
        if col not in df.columns:
            df[col] = None

    # 4. 数值和时间格式转换
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
        errors="coerce"
    )

    # 5. 生成作答事件编号（原字段缺失时）
    missing_event = (
        df["作答事件编号"].isna()
        | (df["作答事件编号"].astype(str).str.strip() == "")
    )

    df.loc[missing_event, "作答事件编号"] = (
        df.loc[missing_event, "学生编号"].astype(str)
        + "-"
        + df.loc[missing_event, "节点编号"].fillna("未知").astype(str)
        + "-"
        + df.loc[missing_event, "材料编号"].astype(str)
    )

    # 6. 保留所有原始版本，只选最新版本作为正式分析主表
    df = df.sort_values(
        ["修订版本", "模拟提交时间"],
        na_position="first"
    )

    latest = df.drop_duplicates(
        subset=["作答事件编号"],
        keep="last"
    ).copy()

    history_count = len(df) - len(latest)

    # 7. 评分状态统计
    status_counts = (
        latest["评分状态"]
        .fillna("状态缺失")
        .astype(str)
        .value_counts()
        .to_dict()
    )

    # 8. 识别有效评分
    latest["评分状态"] = (
        latest["评分状态"].fillna("").astype(str).str.strip()
    )

    latest["有效评分"] = (
        (latest["评分状态"] == "已评分")
        & latest["得分"].notna()
        & latest["满分"].notna()
        & (latest["满分"] > 0)
        & (latest["得分"] >= 0)
        & (latest["得分"] <= latest["满分"])
    )

    # 9. 标记异常记录，不直接删除
    latest["异常原因"] = ""

    latest.loc[
        latest["得分"] < 0,
        "异常原因"
    ] = "得分为负数"

    latest.loc[
        latest["满分"].notna() & (latest["满分"] <= 0),
        "异常原因"
    ] = "满分无效"

    latest.loc[
        latest["得分"] > latest["满分"],
        "异常原因"
    ] = "得分超过满分"

    latest.loc[
        (latest["评分状态"] == "已评分")
        & (latest["得分"].isna() | latest["满分"].isna()),
        "异常原因"
    ] = "已评分但得分或满分缺失"

    latest.loc[
        (latest["评分状态"] == "已评分")
        & latest["模拟提交时间"].isna(),
        "异常原因"
    ] = "提交时间缺失或无法解析"

    # 10. 正式统计数据
    valid = latest[latest["有效评分"]].copy()

    total_score = valid["得分"].sum()
    total_full = valid["满分"].sum()

    overall_rate = (
        round(float(total_score / total_full * 100), 2)
        if total_full > 0 else None
    )

    # 11. 学生个人学情
    student_stats = calculate_group_stats(
        valid, "学生编号"
    )

    # 12. 主题学情
    theme_stats = calculate_group_stats(
        valid, "主题编号"
    )

    # 13. 节点学情
    node_stats = calculate_group_stats(
        valid, "节点编号"
    )

    # 14. 材料学情
    material_stats = calculate_group_stats(
        valid, "材料编号"
    )

    # 15. 可选学生信息表：关联班级
    class_stats = []

    if student_file is not None:
        student_content = await student_file.read()
        students = read_upload(
            student_content,
            student_file.filename
        )

        class_column = next(
            (
                col for col in ["班级", "班级编号", "班级名称"]
                if col in students.columns
            ),
            None
        )

        if "学生编号" in students.columns and class_column:
            roster = students[
                ["学生编号", class_column]
            ].drop_duplicates(subset=["学生编号"])

            class_data = valid.merge(
                roster,
                on="学生编号",
                how="left"
            )

            class_data = class_data.dropna(
                subset=[class_column]
            )

            class_stats = calculate_group_stats(
                class_data, class_column
            )

    # 16. 异常数据清单
    exceptions = latest[
        latest["异常原因"] != ""
    ].copy()

    # 17. 无效评分记录
    invalid_scoring = latest[
        (latest["评分状态"] == "已评分")
        & (~latest["有效评分"])
    ].copy()

    # 18. 数据质量报告
    quality_report = {
        "原始记录数": int(original_count),
        "完全重复记录数": duplicate_count,
        "去重后记录数": int(len(df)),
        "作答事件数": int(len(latest)),
        "历史修订版本数": int(history_count),
        "有效评分数": int(len(valid)),
        "无效评分数": int(len(invalid_scoring)),
        "评分状态分布": status_counts,
        "异常记录数": int(len(exceptions))
    }

    # 19. 输出标准化结果
    return {
        "system": "融学·智伴04统一数据处理模块",
        "version": "2.0.0",
        "filename": file.filename,
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

        "exceptions": records_to_json(
            exceptions.head(100)
        ),

        "invalid_scoring": records_to_json(
            invalid_scoring.head(100)
        ),

        "latest_records_preview": records_to_json(
            latest.head(10)
        )
    }
